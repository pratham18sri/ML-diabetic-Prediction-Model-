import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

const container = document.getElementById("root");
const root = createRoot(container);
root.render(<App />);

console.log("Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)");
