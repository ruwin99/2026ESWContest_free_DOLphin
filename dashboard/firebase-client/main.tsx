import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../app/globals.css";
import { FirebaseDashboard } from "./FirebaseDashboard";
import { firebaseProjectId } from "./firebase";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Dashboard root element was not found.");
}

const webAppHost = firebaseProjectId ? `${firebaseProjectId}.web.app` : "";
if (webAppHost && window.location.hostname === webAppHost) {
  const canonicalUrl = new URL(window.location.href);
  canonicalUrl.hostname = `${firebaseProjectId}.firebaseapp.com`;
  window.location.replace(canonicalUrl.toString());
} else {
  createRoot(root).render(
    <StrictMode>
      <FirebaseDashboard />
    </StrictMode>,
  );
}

