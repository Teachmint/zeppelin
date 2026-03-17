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
 * OptimizerRestApi — Zeppelin REST endpoint that proxies to Vertex AI (Gemini).
 *
 * Configuration is read entirely from the service account JSON key file.
 * The key file path is read from the environment variable:
 *   GOOGLE_APPLICATION_CREDENTIALS
 * If not set, falls back to zeppelin-env.sh or the system default location.
 *
 * Nothing is hardcoded — project, region, model, token_uri, client_email
 * and private_key are all read from the service account JSON at runtime.
 *
 * Registered path: POST /api/optimizer/optimize
 */
@Path("/optimizer")
@Produces(MediaType.APPLICATION_JSON)
public class OptimizerRestApi {

    private static final Logger LOG = LoggerFactory.getLogger(OptimizerRestApi.class);

    // ── Read key file path from environment — nothing hardcoded ──────────────
    // Priority:
    //   1. GOOGLE_APPLICATION_CREDENTIALS env var
    //   2. OPTIMIZER_KEY_FILE env var (Zeppelin-specific override)
    //   3. zeppelin.optimizer.key.file system property
    private static final String KEY_FILE = resolveKeyFile();

    // ── Model and region read from env — fallback to values in key file ───────
    // Set these in zeppelin-env.sh if you want to override:
    //   export VERTEX_MODEL=gemini-2.5-flash
    //   export GCP_REGION=asia-south1
    private static final String GEMINI_MODEL = System.getenv().getOrDefault(
        "AI_MODEL", "gemini-2.5-flash");
    private static final String GCP_REGION_OVERRIDE = System.getenv("area");

    private static final String VERTEX_SCOPE =
        "https://www.googleapis.com/auth/cloud-platform";

    private static final int TIMEOUT_MS = 60_000;

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

    // ── Resolve key file path from environment ────────────────────────────────
    private static String resolveKeyFile() {
        // 1. Standard Google env var
        String path = System.getenv("GPATH");
        if (path != null && !path.trim().isEmpty()) return path.trim();

        // 2. Zeppelin-specific override
        path = System.getenv("GPATH");
        if (path != null && !path.trim().isEmpty()) return path.trim();

        // 3. Java system property
        path = System.getenv("GPATH");
        if (path != null && !path.trim().isEmpty()) return path.trim();

        // 4. No path found — will fail at runtime with a clear error message
        return null;
    }

    // ── POST /api/optimizer/optimize ──────────────────────────────────────────
    @POST
    @Path("/optimize")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response optimize(String requestBody) {

        // 1. Shiro authentication check
        Subject currentUser = SecurityUtils.getSubject();
        if (!currentUser.isAuthenticated()) {
            LOG.warn("Unauthenticated request to /api/optimizer/optimize");
            return new JsonResponse<>(Response.Status.FORBIDDEN,
                "Not authenticated. Please log in to Zeppelin first.").build();
        }

        // 2. Validate key file is configured
        if (KEY_FILE == null) {
            LOG.error("No service account key file configured. " +
                "Set GPATH in zeppelin-env.sh");
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "Optimizer not configured: GPATH is not set. " +
                "Add it to zeppelin-env.sh").build();
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

        String sqlQuery = parsed.get("sql_query").getAsString().trim();
        String explainOutput = parsed.has("explain_output")
            ? parsed.get("explain_output").getAsString().trim() : "";

        if (sqlQuery.length() < 5) {
            return new JsonResponse<>(Response.Status.BAD_REQUEST,
                "sql_query is too short.").build();
        }

        LOG.info("Optimizing SQL for user '{}': {}...",
            currentUser.getPrincipal(),
            sqlQuery.substring(0, Math.min(80, sqlQuery.length())));

        try {
            // 4. Load service account and get token — everything read from JSON
            ServiceAccountInfo sa = loadServiceAccount();
            String accessToken = getAccessToken(sa);
            String vertexUrl = buildVertexUrl(sa.projectId);

            // 5. Call Vertex AI
            String geminiResult = callVertexAI(accessToken, vertexUrl, sqlQuery, explainOutput);

            // 6. Parse and return result
            JsonObject resultJson;
            try {
                String cleaned = geminiResult
                    .replaceAll("(?s)^```json\\s*", "")
                    .replaceAll("(?s)^```\\s*",     "")
                    .replaceAll("(?s)\\s*```$",     "")
                    .trim();
                resultJson = JsonParser.parseString(cleaned).getAsJsonObject();
            } catch (Exception e) {
                LOG.warn("Gemini returned non-JSON response, wrapping as raw_response");
                JsonObject fallback = new JsonObject();
                fallback.addProperty("raw_response", geminiResult);
                return new JsonResponse<>(Response.Status.OK, fallback).build();
            }

            return new JsonResponse<>(Response.Status.OK, resultJson).build();

        } catch (VertexAuthException e) {
            LOG.error("Vertex AI auth failed: {}", e.getMessage());
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "Auth failed: " + e.getMessage()).build();
        } catch (VertexCallException e) {
            LOG.error("Vertex AI call failed ({}): {}", e.statusCode, e.getMessage());
            return new JsonResponse<>(Response.Status.SERVICE_UNAVAILABLE,
                "Vertex AI error (" + e.statusCode + "): " + e.getMessage()).build();
        } catch (IOException e) {
            LOG.error("I/O error: {}", e.getMessage());
            return new JsonResponse<>(Response.Status.INTERNAL_SERVER_ERROR,
                "I/O error: " + e.getMessage()).build();
        }
    }

    // ── Service account info read entirely from JSON key file ─────────────────
    private static class ServiceAccountInfo {
        String clientEmail;   // from key file: client_email
        String privateKeyPem; // from key file: private_key
        String tokenUri;      // from key file: token_uri
        String projectId;     // from key file: project_id
        String region;        // from env GCP_REGION, fallback "us-central1"
    }

    private ServiceAccountInfo loadServiceAccount() throws VertexAuthException {
        try {
            String keyJson = new String(
                Files.readAllBytes(Paths.get(KEY_FILE)), StandardCharsets.UTF_8);
            JsonObject keyObj = JsonParser.parseString(keyJson).getAsJsonObject();

            ServiceAccountInfo sa = new ServiceAccountInfo();
            sa.clientEmail   = keyObj.get("client_email").getAsString();
            sa.privateKeyPem = keyObj.get("private_key").getAsString();
            sa.tokenUri      = keyObj.get("token_uri").getAsString();
            sa.projectId     = keyObj.get("project_id").getAsString();

            // Region: env var takes priority, otherwise default to us-central1
            sa.region = (GCP_REGION_OVERRIDE != null && !GCP_REGION_OVERRIDE.isEmpty())
                ? GCP_REGION_OVERRIDE : "us-central1";

            LOG.info("Loaded service account: {} | project: {} | region: {}",
                sa.clientEmail, sa.projectId, sa.region);
            return sa;

        } catch (IOException e) {
            throw new VertexAuthException(
                "Cannot read service account key from '" + KEY_FILE + "': " + e.getMessage() +
                ". Check that GOOGLE_APPLICATION_CREDENTIALS is set correctly in zeppelin-env.sh");
        } catch (Exception e) {
            throw new VertexAuthException(
                "Invalid service account JSON at '" + KEY_FILE + "': " + e.getMessage());
        }
    }

    // ── Build Vertex AI URL from service account project + env region ─────────
    private String buildVertexUrl(String projectId) {
        String region = (GCP_REGION_OVERRIDE != null && !GCP_REGION_OVERRIDE.isEmpty())
            ? GCP_REGION_OVERRIDE : "us-central1";
        return String.format(
            "https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s" +
            "/publishers/google/models/%s:generateContent",
            region, projectId, region, GEMINI_MODEL
        );
    }

    // ── OAuth2 JWT flow — pure JDK, no external libraries ────────────────────
    // Reads everything from the service account JSON.
    // Does NOT touch gcloud. Does NOT switch active accounts.
    // Has zero side effects on other GCP services.
    private String getAccessToken(ServiceAccountInfo sa) throws VertexAuthException {
        try {
            // 1. Parse RSA private key from PEM
            String pem = sa.privateKeyPem
                .replace("-----BEGIN PRIVATE KEY-----", "")
                .replace("-----END PRIVATE KEY-----", "")
                .replaceAll("\\s", "");
            byte[] keyBytes = Base64.getDecoder().decode(pem);
            PrivateKey privateKey = KeyFactory.getInstance("RSA")
                .generatePrivate(new PKCS8EncodedKeySpec(keyBytes));

            // 2. Build JWT header + claims
            long now = System.currentTimeMillis() / 1000;
            String header = base64url(
                "{\"alg\":\"RS256\",\"typ\":\"JWT\"}".getBytes(StandardCharsets.UTF_8));

            Map<String, Object> claimsMap = new LinkedHashMap<>();
            claimsMap.put("iss",   sa.clientEmail);
            claimsMap.put("scope", VERTEX_SCOPE);
            claimsMap.put("aud",   sa.tokenUri);
            claimsMap.put("iat",   now);
            claimsMap.put("exp",   now + 3600);
            String claims = base64url(gson.toJson(claimsMap).getBytes(StandardCharsets.UTF_8));

            // 3. Sign JWT with RS256
            String signingInput = header + "." + claims;
            Signature signer = Signature.getInstance("SHA256withRSA");
            signer.initSign(privateKey);
            signer.update(signingInput.getBytes(StandardCharsets.UTF_8));
            String jwt = signingInput + "." + base64url(signer.sign());

            // 4. Exchange JWT for OAuth2 access token
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
            conn.getOutputStream().write(postBody.getBytes(StandardCharsets.UTF_8));

            int status = conn.getResponseCode();
            InputStream is = status >= 400 ? conn.getErrorStream() : conn.getInputStream();
            String resp = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            conn.disconnect();

            if (status >= 400) {
                throw new VertexAuthException(
                    "Token endpoint returned " + status + ": " + resp);
            }

            String token = JsonParser.parseString(resp)
                .getAsJsonObject().get("access_token").getAsString();

            LOG.info("Access token obtained via JWT for {} ✅", sa.clientEmail);
            return token;

        } catch (VertexAuthException e) {
            throw e;
        } catch (Exception e) {
            throw new VertexAuthException(
                "JWT auth failed: " + e.getClass().getSimpleName() + " — " + e.getMessage());
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
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setConnectTimeout(TIMEOUT_MS);
            conn.setReadTimeout(TIMEOUT_MS);
            conn.setDoOutput(true);

            byte[] body = gson.toJson(payload).getBytes(StandardCharsets.UTF_8);
            conn.setRequestProperty("Content-Length", String.valueOf(body.length));
            try (OutputStream os = conn.getOutputStream()) { os.write(body); }

            int status = conn.getResponseCode();
            boolean isError = status >= 400;
            InputStream is = isError ? conn.getErrorStream() : conn.getInputStream();
            String responseBody = new String(is.readAllBytes(), StandardCharsets.UTF_8);

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
