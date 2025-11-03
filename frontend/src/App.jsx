import React from "react";
import PredictionForm from "./PredictionForm";
import "./styles.css";

export default function App() {
  return (
    <div className="app-root">
      <header className="app-header">
        <h1>🩺 Diabetes Prediction</h1>
        <p className="lead">Enter patient data below and choose a model to predict diabetes risk.</p>
      </header>

      <main>
        <PredictionForm />
      </main>

      <footer className="app-footer">Built with Python (Flask) + React. Backend: http://127.0.0.1:5000</footer>
    </div>
  );
}
