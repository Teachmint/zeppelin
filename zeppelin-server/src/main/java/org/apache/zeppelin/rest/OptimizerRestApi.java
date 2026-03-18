package org.apache.zeppelin.rest;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.apache.shiro.SecurityUtils;
import org.apache.shiro.subject.Subject;
import org.apache.zeppelin.server.JsonResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.KeyFactory;
import java.security.PrivateKey;
import java.security.Signature;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * OptimizerRestApi — Zeppelin REST endpoint that:
 *   1. Receives sql_query + notebook_id + paragraph_id from the JS button
 *   2. Fetches EXPLAIN FORMATTED output by calling Zeppelin's own paragraph API
 *      (reuses the browser session cookie forwarded from the incoming request)
 *   3. Sends sql + explain to Vertex AI Gemini for optimization
 *   4. Returns structured JSON result to the browser
 *
 * Registered path: POST /api/optimizer/optimize
 */
@Path("/optimizer")
@Produces(MediaType.APPLICATION_JSON)
public class OptimizerRestApi {

    private static final Logger LOG = LoggerFactory.getLogger(OptimizerRestApi.class);

    // ── Environment variables ─────────────────────────────────────────────────
    private static final String KEY_FILE     = resolveKeyFile();
    private static final String GEMINI_MODEL = System.getenv().getOrDefault("AI_MODEL", "gemini-2.5-flash");
    private static final String GCP_REGION   = System.getenv().getOrDefault("area", "us-central1");

    private static final String VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
    private static final int    TIMEOUT_MS   = 60_000;
    private static final int    ZEPPELIN_EXPLAIN_TIMEOUT_MS = 60_000; // 60s for EXPLAIN to finish

    // ── Size limits ───────────────────────────────────────────────────────────
    private static final int MAX_SQL_LENGTH     = 100_000;
    private static final int MAX_EXPLAIN_LENGTH = 500_000;

    // ── Zeppelin internal base URL — loopback, same JVM ──────────────────────
    // Calls go to localhost so no network hop. Port read from env or default 8080.
    // Use ZEPPELIN_ADDR env var if set, otherwise fall back to localhost
    private static final String ZEPPELIN_BASE =
        "http://" + System.getenv().getOrDefault("ZEPPELIN_ADDR", "localhost") +
        ":" + System.getenv().getOrDefault("ZEPPELIN_PORT", "8081");

    // ── Token cache ───────────────────────────────────────────────────────────
    private static volatile ServiceAccountInfo cachedSaInfo      = null;
    private static volatile String             cachedToken       = null;
    private static volatile long               tokenExpiryTimeMs = 0;
    private static final   Object              lock              = new Object();
    private static final   long TOKEN_LIFETIME_MS       = 3600L * 1000L;
    private static final   long TOKEN_REFRESH_BUFFER_MS =  300L * 1000L;

    private static final String SYSTEM_PROMPT =
        "You are an Apache Spark SQL performance expert.\n\n" +
        "You will receive a SQL query and optionally a Spark physical execution plan " +
        "from EXPLAIN FORMATTED. Analyze both and return ONLY a raw JSON object — " +
        "no markdown, no backticks, no explanation text whatsoever.\n\n" +
        "STRICT RULES:\n" +
        "- The optimized SQL MUST return identical rows and columns as the original\n" +
        "- Do NOT change JOIN types unless it is strictly safe\n" +
        "- Do NOT use approximate functions\n" +
        "- Do NOT add LIMIT, sampling, or change aggregation logic\n" +
        "- If the query is already optimal, set is_already_optimal to true\n\n" +
        "Return this exact JSON structure:\n" +
        "{\n" +
        "  \"optimized_sql\": \"the full rewritten SQL here\",\n" +
        "  \"is_already_optimal\": false,\n" +
        "  \"has_cartesian_product\": false,\n" +
        "  \"problems_found\": [{\"type\": \"SHUFFLE|FULL_SCAN|INEFFICIENT_JOIN|SORT|OTHER\"," +
        " \"description\": \"...\", \"location\": \"...\"}],\n" +
        "  \"changes_made\": [{\"what\": \"...\", \"why\": \"...\", \"spark_concept\": \"...\"," +
        " \"impact\": \"HIGH|MEDIUM|LOW\"}],\n" +
        "  \"plan_analysis\": {\"exchange_nodes_before\": 0," +
        " \"exchange_nodes_after_estimate\": 0," +
        " \"join_strategies_before\": [], \"join_strategies_after\": []},\n" +
        "  \"summary\": \"2-3 sentence plain English summary\",\n" +
        "  \"estimated_improvement\": \"e.g. 30-50% faster\"\n" +
        "}";

    private final Gson gson = new Gson();

    // Inject HttpServletRequest so we can forward the session cookie to Zeppelin API
    @Context
    private HttpServletRequest httpRequest;

    private static String resolveKeyFile() {
        String path = System.getenv("GPATH");
        if (path != null && !path.trim().isEmpty()) return path.trim();
        path = System.getProperty("GPATH");
        if (path != null && !path.trim().isEmpty()) return path.trim();
        return null;
    }

    // ── POST /api/optimizer/optimize ──────────────────────────────────────────
    @POST
    @Path("/optimize")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response optimize(String requestBody) {

        // 1. Shiro auth
        Subject currentUser = SecurityUtils.getSubject();
        if (!currentUser.isAuthenticated()) {
            LOG.warn("Unauthenticated request to /api/optimizer/optimize");
            return new JsonResponse<>(Response.Status.FORBIDDEN,
                "Not authenticated. Please log in to Zeppelin first.").build();
        }

        // 2. GPATH configured?
        if (KEY_FILE == null) {
            LOG.error("Optimizer not configured: GPATH is not set.");
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "Optimizer not configured: GPATH is not set in zeppelin-env.sh").build();
        }

        // 3. Parse request body
        if (requestBody == null || requestBody.trim().isEmpty()) {
            return new JsonResponse<>(Response.Status.BAD_REQUEST,
                "Request body must not be empty.").build();
        }

        JsonObject parsed;
        try {
            parsed = JsonParser.parseString(requestBody).getAsJsonObject();
        } catch (Exception e) {
            return new JsonResponse<>(Response.Status.BAD_REQUEST,
                "Invalid JSON: " + e.getMessage()).build();
        }

        if (!parsed.has("sql_query")) {
            return new JsonResponse<>(Response.Status.BAD_REQUEST,
                "Request body must contain 'sql_query'.").build();
        }

        String sqlQuery    = parsed.get("sql_query").getAsString().trim();
        String notebookId  = parsed.has("notebook_id")  ? parsed.get("notebook_id").getAsString().trim()  : "";
        String paragraphId = parsed.has("paragraph_id") ? parsed.get("paragraph_id").getAsString().trim() : "";

        if (sqlQuery.length() < 5) {
            return new JsonResponse<>(Response.Status.BAD_REQUEST,
                "sql_query is too short.").build();
        }

        // 4. Validate — SQL or PySpark only, no prompt injection
        String validationError = validateInput(sqlQuery);
        if (validationError != null) {
            LOG.warn("Input rejected for user '{}': {}", currentUser.getPrincipal(), validationError);
            return new JsonResponse<>(Response.Status.BAD_REQUEST, validationError).build();
        }

        // 5. SQL size check
        if (sqlQuery.length() > MAX_SQL_LENGTH) {
            return new JsonResponse<>(Response.Status.BAD_REQUEST,
                "sql_query exceeds maximum allowed length of " + MAX_SQL_LENGTH + " characters.").build();
        }

        // 6. Fetch EXPLAIN output from Zeppelin API using forwarded session cookie
        // If notebook_id or paragraph_id are missing, skip silently
        String explainOutput  = "";
        boolean explainDropped = false;

        if (!notebookId.isEmpty() && !paragraphId.isEmpty()) {
            LOG.info("═══ EXPLAIN FETCH START ═══");
            LOG.info("notebook_id={} paragraph_id={}", notebookId, paragraphId);
            try {
                String sessionCookie = extractSessionCookie();
                LOG.info("Session cookie present: {}", !sessionCookie.isEmpty());
                LOG.debug("Session cookie value: {}", sessionCookie);
                explainOutput = fetchExplainFromZeppelin(notebookId, sqlQuery, sessionCookie);
                LOG.info("═══ EXPLAIN FETCH SUCCESS ═══ length={} chars", explainOutput.length());
                LOG.debug("Explain output preview:\n{}", explainOutput.substring(0, Math.min(500, explainOutput.length())));
            } catch (Exception e) {
                LOG.warn("═══ EXPLAIN FETCH FAILED ═══ reason={} — optimizing without plan", e.getMessage());
                explainOutput = "";
            }
        } else {
            LOG.info("═══ EXPLAIN FETCH SKIPPED ═══ notebook_id='{}' paragraph_id='{}' — optimizing SQL only",
                notebookId, paragraphId);
        }

        // 7. Drop explain if over size limit
        if (explainOutput.length() > MAX_EXPLAIN_LENGTH) {
            LOG.warn("explain_output exceeds {} chars ({}) — dropping", MAX_EXPLAIN_LENGTH, explainOutput.length());
            explainOutput  = "";
            explainDropped = true;
        }

        LOG.info("Optimizing SQL for user '{}' [explain={}]: {}...",
            currentUser.getPrincipal(),
            explainDropped ? "dropped(too large)" : (explainOutput.isEmpty() ? "none" : "present"),
            sqlQuery.substring(0, Math.min(80, sqlQuery.length())));

        try {
            ServiceAccountInfo sa = getServiceAccountInfo();
            String accessToken    = getValidAccessToken(sa);
            String vertexUrl      = buildVertexUrl(sa.projectId);
            String geminiResult   = callVertexAI(accessToken, vertexUrl, sqlQuery, explainOutput);
            String cleanedJson    = extractJsonFromMarkdown(geminiResult);

            JsonObject resultJson;
            try {
                resultJson = JsonParser.parseString(cleanedJson).getAsJsonObject();
            } catch (Exception e) {
                LOG.warn("Gemini returned non-JSON, wrapping as raw_response.");
                JsonObject fallback = new JsonObject();
                fallback.addProperty("raw_response", geminiResult);
                if (explainDropped) fallback.addProperty("explain_warning",
                    "Execution plan was not used: exceeded " + MAX_EXPLAIN_LENGTH + " char limit.");
                return new JsonResponse<>(Response.Status.OK, fallback).build();
            }

            if (explainDropped) {
                resultJson.addProperty("explain_warning",
                    "Execution plan was not used: exceeded " + MAX_EXPLAIN_LENGTH +
                    " character limit. Optimization is based on SQL structure only.");
            }

            return new JsonResponse<>(Response.Status.OK, resultJson).build();

        } catch (VertexAuthException e) {
            synchronized (lock) { cachedToken = null; tokenExpiryTimeMs = 0; }
            LOG.error("Vertex AI auth failed: {}", e.getMessage());
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "Auth failed: " + e.getMessage()).build();

        } catch (VertexCallException e) {
            LOG.error("Vertex AI call failed ({}): {}", e.statusCode, e.getMessage());
            return new JsonResponse<>(Response.Status.SERVICE_UNAVAILABLE,
                "Vertex AI error (" + e.statusCode + "): " + e.getMessage()).build();

        } catch (IOException e) {
            LOG.error("I/O error: {}", e.getMessage(), e);
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "I/O error: " + e.getMessage()).build();

        } catch (Exception e) {
            LOG.error("Unexpected error", e);
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "Unexpected error: " + e.getMessage()).build();
        }
    }

    // ── Extract session cookie from incoming browser request ──────────────────
    // The browser sends its Zeppelin session cookie with the /api/optimizer/optimize
    // request. We forward it to Zeppelin's internal paragraph API so the call is
    // authenticated as the same user — no separate credentials needed.
    private String extractSessionCookie() {
        javax.servlet.http.Cookie[] cookies = httpRequest.getCookies();
        if (cookies == null) return "";
        StringBuilder sb = new StringBuilder();
        for (javax.servlet.http.Cookie c : cookies) {
            if (sb.length() > 0) sb.append("; ");
            sb.append(c.getName()).append("=").append(c.getValue());
        }
        return sb.toString();
    }

    // ── Fetch EXPLAIN FORMATTED by calling Zeppelin paragraph API internally ──
    // Flow: create temp paragraph → set text → run → poll → get result → delete
    // All HTTP calls go to localhost (same machine) with the forwarded session cookie.
    private String fetchExplainFromZeppelin(String notebookId, String sql, String cookie)
            throws Exception {

        String base = ZEPPELIN_BASE + "/api";
        String pid  = null;

        try {
            // Step 1 — create blank paragraph
            String createUrl  = base + "/notebook/" + notebookId + "/paragraph";
            String createBody = gson.toJson(Map.of("title", "__opt_tmp__", "text", ""));
            String createResp = zeppelinPost(createUrl, createBody, cookie);
            JsonObject createJson = JsonParser.parseString(createResp).getAsJsonObject();

            // 0.11.2 returns body as plain string paragraph ID
            com.google.gson.JsonElement bodyEl = createJson.get("body");
            pid = bodyEl.isJsonPrimitive() ? bodyEl.getAsString() : bodyEl.getAsJsonObject().get("paragraphId").getAsString();
            LOG.info("Created temp paragraph: {}", pid);

            // Step 2 — set text to EXPLAIN FORMATTED <sql>
            String updateUrl  = base + "/notebook/" + notebookId + "/paragraph/" + pid;
            String updateBody = gson.toJson(Map.of("text", "%sql\nEXPLAIN FORMATTED\n" + sql));
            zeppelinPut(updateUrl, updateBody, cookie);

            // Step 3 — run paragraph (Zeppelin 0.11.2)
            String runUrl = base + "/notebook/job/" + notebookId + "/" + pid;
            zeppelinPost(runUrl, "{}", cookie);
            LOG.info("Paragraph {} submitted for execution", pid);

            // Step 4 — poll status until FINISHED or ERROR
            String planText = pollAndExtract(base, notebookId, pid, cookie);
            LOG.info("Explain output fetched: {} chars", planText.length());
            return planText;

        } finally {
            // Step 5 — always delete temp paragraph
            if (pid != null) {
                try {
                    zeppelinDelete(base + "/notebook/" + notebookId + "/paragraph/" + pid, cookie);
                    LOG.info("Temp paragraph {} deleted", pid);
                } catch (Exception e) {
                    LOG.warn("Could not delete temp paragraph {}: {}", pid, e.getMessage());
                }
            }
        }
    }

    // Poll indefinitely every 2s until FINISHED or ERROR — no timeout
    // EXPLAIN on large tables or cold Spark sessions can take several minutes
    private String pollAndExtract(String base, String notebookId, String pid, String cookie)
            throws Exception {
        String jobUrl  = base + "/notebook/job/" + notebookId + "/" + pid;
        String paraUrl = base + "/notebook/" + notebookId + "/paragraph/" + pid;
        int i = 0;

        while (true) {
            Thread.sleep(2000);
            String statusResp = zeppelinGet(jobUrl, cookie);
            LOG.debug("[POLL attempt={}] Raw status response: {}", i, statusResp);
            JsonObject statusJson = JsonParser.parseString(statusResp).getAsJsonObject();
            String status = statusJson.getAsJsonObject("body").get("status").getAsString();
            LOG.info("[POLL attempt={}] paragraph={} status={}", i, pid, status);

            if ("FINISHED".equals(status)) {
                LOG.info("[POLL] FINISHED — fetching full paragraph result");
                String paraResp = zeppelinGet(paraUrl, cookie);
                LOG.debug("[POLL] Full paragraph response: {}",
                    paraResp.substring(0, Math.min(300, paraResp.length())));
                JsonObject paraJson = JsonParser.parseString(paraResp).getAsJsonObject();
                JsonObject body     = paraJson.getAsJsonObject("body");
                JsonObject results  = body.getAsJsonObject("results");
                LOG.info("[POLL] Results code={}", results.get("code").getAsString());
                JsonArray msgs = results.getAsJsonArray("msg");
                LOG.info("[POLL] msg array size={}", msgs.size());
                if (msgs.size() > 0) {
                    String raw     = msgs.get(0).getAsJsonObject().get("data").getAsString();
                    String cleaned = raw.replaceFirst("(?i)^plan\\n", "").trim();
                    LOG.info("[POLL] Plan extracted: {} chars (raw={} chars)", cleaned.length(), raw.length());
                    return cleaned;
                }
                LOG.warn("[POLL] msg array is empty — no plan text returned");
                return "";
            }

            if ("ERROR".equals(status) || "ABORT".equals(status)) {
                LOG.error("[POLL] Paragraph ended with status={}", status);
                throw new Exception("EXPLAIN paragraph ended with status: " + status);
            }

            LOG.info("[POLL attempt={}] Status={} — waiting 2s...", i, status);
            i++;
        }
    }

    // ── Zeppelin HTTP helpers — all forward the session cookie ────────────────
    private String zeppelinPost(String url, String body, String cookie) throws IOException {
        return zeppelinHttp("POST", url, body, cookie);
    }

    private String zeppelinPut(String url, String body, String cookie) throws IOException {
        return zeppelinHttp("PUT", url, body, cookie);
    }

    private String zeppelinGet(String url, String cookie) throws IOException {
        return zeppelinHttp("GET", url, null, cookie);
    }

    private void zeppelinDelete(String url, String cookie) throws IOException {
        zeppelinHttp("DELETE", url, null, cookie);
    }

    private String zeppelinHttp(String method, String url, String body, String cookie)
            throws IOException {
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setRequestMethod(method);
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Cookie", cookie);   // ← forward browser session
        conn.setConnectTimeout(10_000);
        conn.setReadTimeout(ZEPPELIN_EXPLAIN_TIMEOUT_MS);

        if (body != null) {
            conn.setDoOutput(true);
            try (OutputStream os = conn.getOutputStream()) {
                os.write(body.getBytes(StandardCharsets.UTF_8));
            }
        }

        int status = conn.getResponseCode();
        try (InputStream is = status >= 400 ? conn.getErrorStream() : conn.getInputStream()) {
            String resp = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            if (status >= 400) {
                throw new IOException(method + " " + url + " returned " + status + ": " + resp);
            }
            return resp;
        } finally {
            conn.disconnect();
        }
    }

    // ── Input validation ──────────────────────────────────────────────────────
    private static final java.util.regex.Pattern SQL_KEYWORDS = java.util.regex.Pattern.compile(
        "^\\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|MERGE|EXPLAIN|" +
        "SHOW|DESCRIBE|USE|TRUNCATE|CACHE|UNCACHE|REFRESH|ANALYZE|MSCK)\\b",
        java.util.regex.Pattern.CASE_INSENSITIVE);

    private static final java.util.regex.Pattern PYSPARK_KEYWORDS = java.util.regex.Pattern.compile(
        "(spark\\s*\\.|df\\s*\\.|sqlContext\\s*\\.|SparkSession|SparkContext|" +
        "createDataFrame|read\\s*\\.|write\\s*\\.|withColumn|groupBy|agg\\s*\\(|" +
        "filter\\s*\\(|where\\s*\\(|join\\s*\\(|orderBy|sortBy|repartition|" +
        "spark\\.sql\\s*\\()",
        java.util.regex.Pattern.CASE_INSENSITIVE);

    private static final java.util.regex.Pattern INJECTION_PATTERN = java.util.regex.Pattern.compile(
        "(?i)(ignore\\s+(all\\s+)?previous\\s+instructions|forget\\s+(everything|all)|" +
        "you\\s+are\\s+now|act\\s+as\\s+|new\\s+instructions|system\\s*:|" +
        "\\[INST\\]|<</SYS>>|<</INST>>)");

    private String validateInput(String sqlQuery) {
        String trimmed = sqlQuery.trim();
        if (INJECTION_PATTERN.matcher(trimmed).find()) return "Invalid input: prompt injection detected.";
        if (SQL_KEYWORDS.matcher(trimmed).find())      return null;
        if (PYSPARK_KEYWORDS.matcher(trimmed).find())  return null;
        return "Invalid input: only SQL queries or PySpark code are accepted.";
    }

    // ── Extract JSON from markdown fences ─────────────────────────────────────
    private String extractJsonFromMarkdown(String text) {
        String c = text.trim();
        if (c.startsWith("```json")) c = c.substring(7);
        else if (c.startsWith("```")) c = c.substring(3);
        if (c.endsWith("```")) c = c.substring(0, c.length() - 3);
        return c.trim();
    }

    // ── Service account ───────────────────────────────────────────────────────
    private static class ServiceAccountInfo {
        String clientEmail;
        PrivateKey privateKey;
        String tokenUri;
        String projectId;
    }

    private ServiceAccountInfo getServiceAccountInfo() throws VertexAuthException {
        if (cachedSaInfo != null) return cachedSaInfo;
        synchronized (lock) {
            if (cachedSaInfo != null) return cachedSaInfo;
            try {
                String keyJson = new String(Files.readAllBytes(Paths.get(KEY_FILE)), StandardCharsets.UTF_8);
                JsonObject k   = JsonParser.parseString(keyJson).getAsJsonObject();
                ServiceAccountInfo sa = new ServiceAccountInfo();
                sa.clientEmail = k.get("client_email").getAsString();
                sa.tokenUri    = k.get("token_uri").getAsString();
                sa.projectId   = k.get("project_id").getAsString();
                String pem = k.get("private_key").getAsString()
                    .replace("-----BEGIN PRIVATE KEY-----", "")
                    .replace("-----END PRIVATE KEY-----", "")
                    .replaceAll("\\s", "");
                sa.privateKey = KeyFactory.getInstance("RSA")
                    .generatePrivate(new PKCS8EncodedKeySpec(Base64.getDecoder().decode(pem)));
                LOG.info("Loaded SA: {} project: {}", sa.clientEmail, sa.projectId);
                cachedSaInfo = sa;
                return sa;
            } catch (IOException e) {
                throw new VertexAuthException("Cannot read GPATH='" + KEY_FILE + "': " + e.getMessage());
            } catch (Exception e) {
                throw new VertexAuthException("Invalid SA JSON at '" + KEY_FILE + "': " + e.getMessage());
            }
        }
    }

    private String buildVertexUrl(String projectId) {
        return String.format(
            "https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s" +
            "/publishers/google/models/%s:generateContent",
            GCP_REGION, projectId, GCP_REGION, GEMINI_MODEL);
    }

    private String getValidAccessToken(ServiceAccountInfo sa) throws VertexAuthException {
        long now = System.currentTimeMillis();
        if (cachedToken != null && now < (tokenExpiryTimeMs - TOKEN_REFRESH_BUFFER_MS)) return cachedToken;
        synchronized (lock) {
            if (cachedToken != null && now < (tokenExpiryTimeMs - TOKEN_REFRESH_BUFFER_MS)) return cachedToken;
            cachedToken       = fetchNewAccessToken(sa);
            tokenExpiryTimeMs = System.currentTimeMillis() + TOKEN_LIFETIME_MS;
            LOG.info("Token refreshed, valid 1hr.");
            return cachedToken;
        }
    }

    private String fetchNewAccessToken(ServiceAccountInfo sa) throws VertexAuthException {
        try {
            long now    = System.currentTimeMillis() / 1000;
            String hdr  = base64url("{\"alg\":\"RS256\",\"typ\":\"JWT\"}".getBytes(StandardCharsets.UTF_8));
            Map<String, Object> claims = new LinkedHashMap<>();
            claims.put("iss", sa.clientEmail); claims.put("scope", VERTEX_SCOPE);
            claims.put("aud", sa.tokenUri);    claims.put("iat", now); claims.put("exp", now + 3600);
            String clm  = base64url(gson.toJson(claims).getBytes(StandardCharsets.UTF_8));
            String si   = hdr + "." + clm;
            Signature sg = Signature.getInstance("SHA256withRSA");
            sg.initSign(sa.privateKey);
            sg.update(si.getBytes(StandardCharsets.UTF_8));
            String jwt  = si + "." + base64url(sg.sign());
            String post = "grant_type=" + java.net.URLEncoder.encode("urn:ietf:params:oauth:grant-type:jwt-bearer", "UTF-8")
                        + "&assertion=" + java.net.URLEncoder.encode(jwt, "UTF-8");
            HttpURLConnection c = (HttpURLConnection) new URL(sa.tokenUri).openConnection();
            c.setRequestMethod("POST");
            c.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
            c.setConnectTimeout(10_000); c.setReadTimeout(10_000); c.setDoOutput(true);
            try (OutputStream os = c.getOutputStream()) { os.write(post.getBytes(StandardCharsets.UTF_8)); }
            int st = c.getResponseCode();
            String resp;
            try (InputStream is = st >= 400 ? c.getErrorStream() : c.getInputStream()) {
                resp = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            } finally { c.disconnect(); }
            if (st >= 400) throw new VertexAuthException("Token endpoint " + st + ": " + resp);
            return JsonParser.parseString(resp).getAsJsonObject().get("access_token").getAsString();
        } catch (VertexAuthException e) { throw e; }
        catch (Exception e) { throw new VertexAuthException("JWT auth failed: " + e.getMessage()); }
    }

    private static String base64url(byte[] d) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(d);
    }

    // ── Vertex AI call ────────────────────────────────────────────────────────
    private String callVertexAI(String token, String url, String sql, String explain)
            throws VertexCallException, IOException {

        StringBuilder msg = new StringBuilder("SQL Query to optimize:\n").append(sql);
        if (explain != null && explain.length() > 50)
            msg.append("\n\nSpark Physical Execution Plan:\n").append(explain);
        else
            msg.append("\n\nNo execution plan — optimize based on SQL structure only.");
        msg.append("\n\nReturn ONLY raw JSON. No markdown. No backticks.");

        JsonObject sysPart = new JsonObject(); sysPart.addProperty("text", SYSTEM_PROMPT);
        JsonArray  sysParts = new JsonArray(); sysParts.add(sysPart);
        JsonObject sysInst = new JsonObject(); sysInst.add("parts", sysParts);

        JsonObject uPart = new JsonObject(); uPart.addProperty("text", msg.toString());
        JsonArray  uParts = new JsonArray(); uParts.add(uPart);
        JsonObject uCont = new JsonObject(); uCont.addProperty("role", "user"); uCont.add("parts", uParts);
        JsonArray  conts = new JsonArray(); conts.add(uCont);

        JsonObject genCfg = new JsonObject();
        genCfg.addProperty("temperature", 0.1);
        genCfg.addProperty("maxOutputTokens", 16384);
        genCfg.addProperty("topP", 0.8);

        JsonObject payload = new JsonObject();
        payload.add("systemInstruction", sysInst);
        payload.add("contents", conts);
        payload.add("generationConfig", genCfg);

        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(url).openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Authorization", "Bearer " + token);
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setConnectTimeout(TIMEOUT_MS); conn.setReadTimeout(TIMEOUT_MS);
            conn.setDoOutput(true);
            byte[] body = gson.toJson(payload).getBytes(StandardCharsets.UTF_8);
            conn.setRequestProperty("Content-Length", String.valueOf(body.length));
            try (OutputStream os = conn.getOutputStream()) { os.write(body); }
            int st = conn.getResponseCode();
            String resp;
            try (InputStream is = st >= 400 ? conn.getErrorStream() : conn.getInputStream()) {
                resp = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            }
            if (st >= 400) throw new VertexCallException(st, resp);
            return JsonParser.parseString(resp).getAsJsonObject()
                .getAsJsonArray("candidates")
                .get(0).getAsJsonObject().getAsJsonObject("content")
                .getAsJsonArray("parts").get(0).getAsJsonObject()
                .get("text").getAsString();
        } finally { if (conn != null) conn.disconnect(); }
    }

    // ── Custom exceptions ─────────────────────────────────────────────────────
    private static class VertexAuthException extends Exception {
        VertexAuthException(String m) { super(m); }
    }
    private static class VertexCallException extends Exception {
        final int statusCode;
        VertexCallException(int c, String b) { super(b); this.statusCode = c; }
    }
}
