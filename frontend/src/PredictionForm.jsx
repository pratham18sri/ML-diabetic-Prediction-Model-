import React, { useState } from "react";

const initial = {
  Pregnancies: 0,
  Glucose: 120,
  BloodPressure: 70,
  SkinThickness: 20,
  Insulin: 80,
  BMI: 25,
  DiabetesPedigreeFunction: 0.5,
  Age: 30,
};

export default function PredictionForm() {
  const [form, setForm] = useState(initial);
  const [model, setModel] = useState("logreg");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const name = e.target.name;
    const value = e.target.value;
    setForm((s) => ({ ...s, [name]: value }));
  };

  const handleModelChange = (e) => {
    setModel(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    // Build payload with numeric values
    const payload = {};
    Object.keys(form).forEach((k) => {
      const v = form[k];
      payload[k] = v === "" ? null : Number(v);
    });
    payload.model = model;

    try {
  const apiHost = window.location.hostname;
  const res = await fetch(`http://${apiHost}:5000/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          {Object.keys(form).map((key) => (
            <label key={key} className="form-row">
              <span className="label">{key}</span>
              <input
                name={key}
                value={form[key]}
                onChange={handleChange}
                type="number"
                step="any"
                className="input"
              />
            </label>
          ))}
        </div>

        <div className="model-select">
          <label>
            <input type="radio" name="model" value="logreg" checked={model === "logreg"} onChange={handleModelChange} /> Logistic Regression
          </label>
          <label>
            <input type="radio" name="model" value="nb" checked={model === "nb"} onChange={handleModelChange} /> Naive Bayes
          </label>
        </div>

        <div style={{ marginTop: 12 }}>
          <button type="submit" className="btn" disabled={loading}>{loading ? "Predicting..." : "Predict"}</button>
        </div>
      </form>

      {result && (
        <div className="result">
          {result.error ? (
            <div className="error">Error: {result.error}</div>
          ) : (
            <>
              <h3 className={result.prediction_binary === 1 ? "warn" : "ok"}>{result.prediction}</h3>
              <p>Confidence: {result.confidence ?? "N/A"}%</p>
              <p className="muted">Model: {result.model}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
