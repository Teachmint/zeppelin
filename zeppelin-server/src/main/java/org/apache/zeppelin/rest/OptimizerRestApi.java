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

import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
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
 * OptimizerRestApi — Zeppelin REST endpoint that calls Vertex AI (Gemini) directly.
 *
 * Configuration via environment variables in zeppelin-env.sh:
 *   GPATH     — path to GCP service account JSON key file (required)
 *   area      — GCP region (optional, default: us-central1)
 *   AI_MODEL  — Vertex AI model name (optional, default: gemini-2.5-flash)
 *
 * Nothing is hardcoded. Project ID, client email, private key and token URI
 * are all read from the service account JSON at runtime.
 *
 * Auth uses pure JDK JWT flow — no Google Auth library, no gcloud,
 * no side effects on other GCP services running on this VM.
 *
 * Registered path: POST /api/optimizer/optimize
 */
@Path("/optimizer")
@Produces(MediaType.APPLICATION_JSON)
public class OptimizerRestApi {

    private static final Logger LOG = LoggerFactory.getLogger(OptimizerRestApi.class);

    // ── Environment variable names ────────────────────────────────────────────
    private static final String KEY_FILE    = resolveKeyFile();
    private static final String GEMINI_MODEL = System.getenv().getOrDefault("AI_MODEL", "gemini-2.5-flash");
    private static final String GCP_REGION   = System.getenv().getOrDefault("area", "us-central1");

    private static final String VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
    private static final int    TIMEOUT_MS   = 60_000;

    // ── Size limits to prevent OOM and API rejections ─────────────────────────
    private static final int MAX_SQL_LENGTH     = 100_000;
    private static final int MAX_EXPLAIN_LENGTH = 500_000;

    // ── Token caching — shared across requests, refreshed before expiry ───────
    private static volatile ServiceAccountInfo cachedSaInfo       = null;
    private static volatile String             cachedToken        = null;
    private static volatile long               tokenExpiryTimeMs  = 0;
    private static final   Object              lock               = new Object();
    private static final   long TOKEN_LIFETIME_MS      = 3600L * 1000L; // 1 hour
    private static final   long TOKEN_REFRESH_BUFFER_MS =  300L * 1000L; // 5 min buffer

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

    // ── Resolve key file path from GPATH env var ──────────────────────────────
    private static String resolveKeyFile() {
        // 1. GPATH environment variable (set in zeppelin-env.sh)
        String path = System.getenv("GPATH");
        if (path != null && !path.trim().isEmpty()) return path.trim();

        // 2. Java system property fallback (-DGPATH=...)
        path = System.getProperty("GPATH");
        if (path != null && !path.trim().isEmpty()) return path.trim();

        return null; // Will trigger clear error message at runtime
    }

    // ── POST /api/optimizer/optimize ──────────────────────────────────────────
    @POST
    @Path("/optimize")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response optimize(String requestBody) {

        // 1. Shiro authentication — Layer 2 (Layer 1 is the Shiro filter)
        Subject currentUser = SecurityUtils.getSubject();
        if (!currentUser.isAuthenticated()) {
            LOG.warn("Unauthenticated request to /api/optimizer/optimize");
            return new JsonResponse<>(Response.Status.FORBIDDEN,
                "Not authenticated. Please log in to Zeppelin first.").build();
        }

        // 2. Check GPATH is configured
        if (KEY_FILE == null) {
            LOG.error("Optimizer not configured: GPATH environment variable is not set.");
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "Optimizer not configured: GPATH is not set in zeppelin-env.sh").build();
        }

        // 3. Validate request body
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

        String sqlQuery      = parsed.get("sql_query").getAsString().trim();
        String explainOutput = parsed.has("explain_output")
            ? parsed.get("explain_output").getAsString().trim() : "";

        if (sqlQuery.length() < 5) {
            return new JsonResponse<>(Response.Status.BAD_REQUEST,
                "sql_query is too short.").build();
        }

        // FIX 2: Use BAD_REQUEST (400) — REQUEST_ENTITY_TOO_LARGE does not exist
        // in javax.ws.rs.core.Response.Status and causes a compile error
        if (sqlQuery.length() > MAX_SQL_LENGTH || explainOutput.length() > MAX_EXPLAIN_LENGTH) {
            return new JsonResponse<>(Response.Status.BAD_REQUEST,
                "Query or explain output exceeds size limits. " +
                "Max SQL: " + MAX_SQL_LENGTH + " chars, " +
                "Max EXPLAIN: " + MAX_EXPLAIN_LENGTH + " chars.").build();
        }

        LOG.info("Optimizing SQL for user '{}': {}...",
            currentUser.getPrincipal(),
            sqlQuery.substring(0, Math.min(80, sqlQuery.length())));

        try {
            ServiceAccountInfo sa  = getServiceAccountInfo();
            String accessToken     = getValidAccessToken(sa);
            String vertexUrl       = buildVertexUrl(sa.projectId);
            String geminiResult    = callVertexAI(accessToken, vertexUrl, sqlQuery, explainOutput);
            String cleanedJson     = extractJsonFromMarkdown(geminiResult);

            JsonObject resultJson;
            try {
                resultJson = JsonParser.parseString(cleanedJson).getAsJsonObject();
            } catch (Exception e) {
                LOG.warn("Gemini returned non-JSON response, wrapping as raw_response.");
                JsonObject fallback = new JsonObject();
                fallback.addProperty("raw_response", geminiResult);
                return new JsonResponse<>(Response.Status.OK, fallback).build();
            }

            return new JsonResponse<>(Response.Status.OK, resultJson).build();

        } catch (VertexAuthException e) {
            // FIX 1: Clear token cache on auth failure so next request forces refresh
            synchronized (lock) {
                cachedToken       = null;
                tokenExpiryTimeMs = 0;
            }
            LOG.error("Vertex AI auth failed, token cache cleared: {}", e.getMessage());
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "Auth failed: " + e.getMessage()).build();

        } catch (VertexCallException e) {
            LOG.error("Vertex AI call failed ({}): {}", e.statusCode, e.getMessage());
            return new JsonResponse<>(Response.Status.SERVICE_UNAVAILABLE,
                "Vertex AI error (" + e.statusCode + "): " + e.getMessage()).build();

        } catch (IOException e) {
            LOG.error("I/O error during optimization: {}", e.getMessage(), e);
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "I/O error: " + e.getMessage()).build();

        } catch (Exception e) {
            LOG.error("Unexpected error during optimization", e);
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "Unexpected error: " + e.getMessage()).build();
        }
    }

    // ── Extract JSON from markdown fences if present ──────────────────────────
    private String extractJsonFromMarkdown(String text) {
        String cleaned = text.trim();
        if (cleaned.startsWith("```json")) {
            cleaned = cleaned.substring(7);
        } else if (cleaned.startsWith("```")) {
            cleaned = cleaned.substring(3);
        }
        if (cleaned.endsWith("```")) {
            cleaned = cleaned.substring(0, cleaned.length() - 3);
        }
        return cleaned.trim();
    }

    // ── Service account — loaded and cached once ──────────────────────────────
    private static class ServiceAccountInfo {
        String     clientEmail;
        PrivateKey privateKey;  // RSA key parsed once and held in memory
        String     tokenUri;
        String     projectId;
    }

    private ServiceAccountInfo getServiceAccountInfo() throws VertexAuthException {
        if (cachedSaInfo != null) return cachedSaInfo;

        synchronized (lock) {
            if (cachedSaInfo != null) return cachedSaInfo;

            try {
                String keyJson = new String(
                    Files.readAllBytes(Paths.get(KEY_FILE)), StandardCharsets.UTF_8);
                JsonObject keyObj = JsonParser.parseString(keyJson).getAsJsonObject();

                ServiceAccountInfo sa = new ServiceAccountInfo();
                sa.clientEmail = keyObj.get("client_email").getAsString();
                sa.tokenUri    = keyObj.get("token_uri").getAsString();
                sa.projectId   = keyObj.get("project_id").getAsString();

                // Parse RSA private key once — expensive operation, cache the result
                String pem = keyObj.get("private_key").getAsString()
                    .replace("-----BEGIN PRIVATE KEY-----", "")
                    .replace("-----END PRIVATE KEY-----", "")
                    .replaceAll("\\s", "");
                byte[] keyBytes = Base64.getDecoder().decode(pem);
                sa.privateKey = KeyFactory.getInstance("RSA")
                    .generatePrivate(new PKCS8EncodedKeySpec(keyBytes));

                LOG.info("Loaded service account: {} | project: {} | region: {} | model: {}",
                    sa.clientEmail, sa.projectId, GCP_REGION, GEMINI_MODEL);
                cachedSaInfo = sa;
                return sa;

            } catch (IOException e) {
                throw new VertexAuthException(
                    "Cannot read service account key from GPATH='" + KEY_FILE +
                    "': " + e.getMessage());
            } catch (Exception e) {
                throw new VertexAuthException(
                    "Invalid service account JSON at '" + KEY_FILE +
                    "': " + e.getMessage());
            }
        }
    }

    // ── Build Vertex AI URL from project ID + region + model ─────────────────
    private String buildVertexUrl(String projectId) {
        return String.format(
            "https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s" +
            "/publishers/google/models/%s:generateContent",
            GCP_REGION, projectId, GCP_REGION, GEMINI_MODEL
        );
    }

    // ── Token management — cached, refreshed 5 min before expiry ─────────────
    private String getValidAccessToken(ServiceAccountInfo sa) throws VertexAuthException {
        long now = System.currentTimeMillis();

        // Fast path — return cached token if still valid
        if (cachedToken != null && now < (tokenExpiryTimeMs - TOKEN_REFRESH_BUFFER_MS)) {
            return cachedToken;
        }

        // Slow path — fetch new token inside lock
        synchronized (lock) {
            // Double-check after acquiring lock
            if (cachedToken != null && now < (tokenExpiryTimeMs - TOKEN_REFRESH_BUFFER_MS)) {
                return cachedToken;
            }
            cachedToken       = fetchNewAccessToken(sa);
            tokenExpiryTimeMs = System.currentTimeMillis() + TOKEN_LIFETIME_MS;
            LOG.info("Token cache refreshed, valid for 1 hour.");
            return cachedToken;
        }
    }

    // ── OAuth2 JWT flow — pure JDK, no external libraries ────────────────────
    // Reads private key from service account JSON.
    // Does NOT touch gcloud. Does NOT switch active accounts.
    // Zero side effects on other GCP services on this VM.
    private String fetchNewAccessToken(ServiceAccountInfo sa) throws VertexAuthException {
        try {
            long now    = System.currentTimeMillis() / 1000;
            String header = base64url(
                "{\"alg\":\"RS256\",\"typ\":\"JWT\"}".getBytes(StandardCharsets.UTF_8));

            Map<String, Object> claimsMap = new LinkedHashMap<>();
            claimsMap.put("iss",   sa.clientEmail);
            claimsMap.put("scope", VERTEX_SCOPE);
            claimsMap.put("aud",   sa.tokenUri);
            claimsMap.put("iat",   now);
            claimsMap.put("exp",   now + 3600);
            String claims = base64url(gson.toJson(claimsMap).getBytes(StandardCharsets.UTF_8));

            String signingInput = header + "." + claims;
            Signature signer = Signature.getInstance("SHA256withRSA");
            signer.initSign(sa.privateKey);
            signer.update(signingInput.getBytes(StandardCharsets.UTF_8));
            String jwt = signingInput + "." + base64url(signer.sign());

            String postBody =
                "grant_type=" + java.net.URLEncoder.encode(
                    "urn:ietf:params:oauth:grant-type:jwt-bearer", "UTF-8") +
                "&assertion=" + java.net.URLEncoder.encode(jwt, "UTF-8");

            HttpURLConnection conn = (HttpURLConnection) new URL(sa.tokenUri).openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
            conn.setConnectTimeout(10_000);
            conn.setReadTimeout(10_000);
            conn.setDoOutput(true);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(postBody.getBytes(StandardCharsets.UTF_8));
            }

            int status = conn.getResponseCode();
            String resp;
            // FIX 3: InputStream wrapped in try-with-resources to prevent leak
            try (InputStream is = status >= 400
                    ? conn.getErrorStream()
                    : conn.getInputStream()) {
                resp = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            } finally {
                conn.disconnect();
            }

            if (status >= 400) {
                throw new VertexAuthException(
                    "Token endpoint returned " + status + ": " + resp);
            }

            LOG.info("New access token obtained via JWT for {} ✅", sa.clientEmail);
            return JsonParser.parseString(resp)
                .getAsJsonObject().get("access_token").getAsString();

        } catch (VertexAuthException e) {
            throw e;
        } catch (Exception e) {
            throw new VertexAuthException(
                "JWT auth failed: " + e.getClass().getSimpleName() +
                " — " + e.getMessage());
        }
    }

    private static String base64url(byte[] data) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(data);
    }

    // ── Call Vertex AI Gemini REST API ────────────────────────────────────────
    private String callVertexAI(String accessToken, String vertexUrl,
                                 String sql, String explain)
            throws VertexCallException, IOException {

        StringBuilder userMsg = new StringBuilder();
        userMsg.append("SQL Query to optimize:\n").append(sql);
        if (explain != null && explain.length() > 50) {
            userMsg.append("\n\nSpark Physical Execution Plan:\n").append(explain);
        } else {
            userMsg.append("\n\nNo execution plan — optimize based on SQL structure only.");
        }
        userMsg.append("\n\nReturn ONLY raw JSON. No markdown. No backticks.");

        JsonObject sysPart = new JsonObject();
        sysPart.addProperty("text", SYSTEM_PROMPT);
        JsonArray sysParts = new JsonArray();
        sysParts.add(sysPart);
        JsonObject systemInstruction = new JsonObject();
        systemInstruction.add("parts", sysParts);

        JsonObject userPart = new JsonObject();
        userPart.addProperty("text", userMsg.toString());
        JsonArray userParts = new JsonArray();
        userParts.add(userPart);
        JsonObject userContent = new JsonObject();
        userContent.addProperty("role", "user");
        userContent.add("parts", userParts);
        JsonArray contents = new JsonArray();
        contents.add(userContent);

        JsonObject generationConfig = new JsonObject();
        generationConfig.addProperty("temperature",     0.1);
        generationConfig.addProperty("maxOutputTokens", 16384);
        generationConfig.addProperty("topP",            0.8);

        JsonObject payload = new JsonObject();
        payload.add("systemInstruction", systemInstruction);
        payload.add("contents",          contents);
        payload.add("generationConfig",  generationConfig);

        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(vertexUrl).openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Authorization", "Bearer " + accessToken);
            conn.setRequestProperty("Content-Type",  "application/json; charset=UTF-8");
            conn.setConnectTimeout(TIMEOUT_MS);
            conn.setReadTimeout(TIMEOUT_MS);
            conn.setDoOutput(true);

            byte[] body = gson.toJson(payload).getBytes(StandardCharsets.UTF_8);
            conn.setRequestProperty("Content-Length", String.valueOf(body.length));
            try (OutputStream os = conn.getOutputStream()) {
                os.write(body);
            }

            int status = conn.getResponseCode();
            boolean isError = status >= 400;

            // FIX 3 applied here too — InputStream in try-with-resources
            String responseBody;
            try (InputStream is = isError
                    ? conn.getErrorStream()
                    : conn.getInputStream()) {
                responseBody = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            }

            if (isError) throw new VertexCallException(status, responseBody);

            JsonObject resp = JsonParser.parseString(responseBody).getAsJsonObject();
            return resp.getAsJsonArray("candidates")
                       .get(0).getAsJsonObject()
                       .getAsJsonObject("content")
                       .getAsJsonArray("parts")
                       .get(0).getAsJsonObject()
                       .get("text").getAsString();

        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    // ── Custom exceptions ─────────────────────────────────────────────────────
    private static class VertexAuthException extends Exception {
        VertexAuthException(String msg) { super(msg); }
    }

    private static class VertexCallException extends Exception {
        final int statusCode;
        VertexCallException(int code, String body) {
            super(body);
            this.statusCode = code;
        }
    }
}
