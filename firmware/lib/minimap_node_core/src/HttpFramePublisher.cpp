#include "mmpr/HttpFramePublisher.h"

#include <WiFi.h>

#include "mmpr/NodeProtocol.h"

namespace mmpr {

HttpFramePublisher::HttpFramePublisher(const char* serverBaseUrl, const char* ingestPath, uint32_t timeoutMs)
    : timeoutMs_(timeoutMs) {
  endpointUrl_ = String(serverBaseUrl != nullptr ? serverBaseUrl : "") +
                 String(ingestPath != nullptr ? ingestPath : "/api/v1/ingest/frame");
  endpointValid_ = parseEndpoint();
}

bool HttpFramePublisher::parseEndpoint() {
  host_ = String();
  path_ = String();
  port_ = 80;

  String url = endpointUrl_;
  url.trim();

  if (!url.startsWith("http://")) {
    return false;
  }

  url.remove(0, 7);  // Strip "http://"

  int pathStart = url.indexOf('/');
  String hostPort = (pathStart >= 0) ? url.substring(0, pathStart) : url;
  path_ = (pathStart >= 0) ? url.substring(pathStart) : String("/");

  if (hostPort.length() == 0) {
    return false;
  }

  const int colonIndex = hostPort.lastIndexOf(':');
  if (colonIndex > 0) {
    host_ = hostPort.substring(0, colonIndex);
    const String portText = hostPort.substring(colonIndex + 1);
    const long parsedPort = portText.toInt();
    if (parsedPort <= 0 || parsedPort > 65535) {
      return false;
    }
    port_ = static_cast<uint16_t>(parsedPort);
  } else {
    host_ = hostPort;
  }

  if (host_.length() == 0) {
    return false;
  }

  if (!path_.startsWith("/")) {
    path_ = String("/") + path_;
  }

  return true;
}

PublishResult HttpFramePublisher::publish(const NodeDescriptor& node, const AudioFrame& frame, bool keepResponseBody) {
  PublishResult result = {false, -1, String()};

  if (!endpointValid_) {
    result.statusCode = -3;
    return result;
  }

  if (WiFi.status() != WL_CONNECTED) {
    result.statusCode = -1;
    return result;
  }

  String payload;
  if (!buildIngestPayload(node, frame, payload)) {
    result.statusCode = -2;
    return result;
  }

  WiFiClient client;
  client.setTimeout(timeoutMs_);

  if (!client.connect(host_.c_str(), port_)) {
    result.statusCode = -3;
    client.stop();
    return result;
  }

  client.print("POST ");
  client.print(path_);
  client.print(" HTTP/1.1\r\nHost: ");
  client.print(host_);
  if (port_ != 80) {
    client.print(':');
    client.print(port_);
  }
  client.print("\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: ");
  client.print(payload.length());
  client.print("\r\n\r\n");
  client.print(payload);

  const uint32_t startMs = millis();
  while (!client.available() && (millis() - startMs) < timeoutMs_) {
    delay(1);
  }

  if (!client.available()) {
    result.statusCode = -4;
    client.stop();
    return result;
  }

  const String statusLine = client.readStringUntil('\n');
  int statusCode = -4;

  const int firstSpace = statusLine.indexOf(' ');
  if (firstSpace > 0 && statusLine.startsWith("HTTP/")) {
    const int secondSpace = statusLine.indexOf(' ', firstSpace + 1);
    const String codeText = (secondSpace > firstSpace)
                                ? statusLine.substring(firstSpace + 1, secondSpace)
                                : statusLine.substring(firstSpace + 1);
    statusCode = codeText.toInt();
  }

  // Consume headers.
  while (client.connected()) {
    const String line = client.readStringUntil('\n');
    if (line == "\r" || line.length() == 0) {
      break;
    }
  }

  if (keepResponseBody || statusCode >= 300 || statusCode < 0) {
    const uint32_t bodyStartMs = millis();
    while ((client.connected() || client.available()) && (millis() - bodyStartMs) < timeoutMs_) {
      while (client.available()) {
        result.responseBody += static_cast<char>(client.read());
      }
      delay(1);
    }
  }

  client.stop();

  result.statusCode = statusCode;
  result.ok = (statusCode >= 200 && statusCode < 300);
  return result;
}

}  // namespace mmpr
