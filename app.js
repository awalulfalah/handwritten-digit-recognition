import http from "http";
import { readFile } from "fs/promises";
import { extname, resolve, join, normalize, relative } from "path";
import mimeTypes from "./constants/mime-types.js";
import SERVER_CONFIG from "./constants/server-config.js";

let cached404Content = null;
let is404ContentLoaded = false;

const server = http.createServer(async (req, res) => {
  const publicDir = resolve("public");

  const url = new URL(req.url, `http://${req.headers.host}`);
  let pathname = url.pathname === "/" ? "/index.html" : url.pathname;
  
  const filePath = join(publicDir, pathname);
  
  const normalizedPath = normalize(filePath);
  const relativePath = relative(publicDir, normalizedPath);
  
  if (relativePath.startsWith("..")) {
    await send404Response(res, publicDir);
    return;
  }

  const ext = extname(filePath);
  const contentType = mimeTypes[ext] || "application/octet-stream";
  
  const finalContentType = contentType.startsWith("text/") || 
                          contentType === "application/javascript" || 
                          contentType === "application/json"
                          ? `${contentType}; charset=utf-8`
                          : contentType;

  try {
    const content = await readFile(filePath, {
      encoding: contentType.startsWith("text/") ? "utf-8" : null,
    });
    
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("X-Frame-Options", "DENY");
    
    if ([".jpg", ".jpeg", ".png", ".gif", ".ico", ".css", ".js"].includes(ext)) {
      res.setHeader("Cache-Control", "public, max-age=86400");
    } else {
      res.setHeader("Cache-Control", "no-cache");
    }
    
    res.writeHead(200, { "Content-Type": finalContentType });
    res.end(content);
  } catch (error) {
    if (error.code === "ENOENT") {
      await send404Response(res, publicDir);
    } else {
      console.error("Server error:", error);
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end("Internal Server Error");
    }
  }
});

const { HOST, PORT } = SERVER_CONFIG;

server.listen(PORT, HOST, (err) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`Server running on http://${HOST}:${PORT}`);
});

process.on("SIGINT", () => {
  console.log("\nShutting down server...");
  server.close(() => {
    process.exit(0);
  });
});

/**
 * @param {http.ServerResponse} res - The response object
 * @param {string} publicDir - Path to public directory
 */
async function send404Response(res, publicDir) {
  if (!is404ContentLoaded) {
    try {
      cached404Content = await readFile(resolve(publicDir, "404.html"), "utf-8");
    } catch (error) {
      console.warn("Warning: 404.html not found or corrupted, using fallback");
      cached404Content = null;
    }
    is404ContentLoaded = true;
  }

  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("X-Frame-Options", "DENY");
  res.setHeader("Cache-Control", "no-cache");

  if (cached404Content) {
    res.writeHead(404, { "Content-Type": "text/html; charset=utf-8" });
    res.end(cached404Content);
  } else {
    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("Not Found");
  }
}
