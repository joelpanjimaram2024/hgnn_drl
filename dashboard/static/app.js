(function () {
  "use strict";

  const form = document.getElementById("run-form");
  const runBtn = document.getElementById("run-btn");
  const runStatus = document.getElementById("run-status");
  const resultsCard = document.getElementById("results-card");
  const resultsHint = document.getElementById("results-hint");
  const resultsTable = document.getElementById("results-table");
  const compareCard = document.getElementById("compare-card");
  const compareHint = document.getElementById("compare-hint");
  const chartCanvas = document.getElementById("chart");

  let chart = null;
  const baselines = ["cloud-only", "greedy", "mist-only"];

  function setControlsDisabled(disabled) {
    form.querySelectorAll("select, input").forEach((el) => (el.disabled = disabled));
    runBtn.disabled = disabled;
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return res.json();
  }

  async function getJson(url) {
    const res = await fetch(url);
    return res.json();
  }

  function showError(el, msg) {
    el.textContent = msg;
    el.style.color = "var(--danger)";
  }

  function showHint(el, msg) {
    el.textContent = msg;
    el.style.color = "var(--muted)";
  }

  function fmt(v, digits = 4) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    return Number(v).toFixed(digits);
  }

  function renderTable(rows) {
    if (!rows.length) {
      resultsHint.textContent = "No result rows returned.";
      return;
    }
    const cols = Object.keys(rows[0]);
    let html = "<tr><th>metric</th>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
    for (const row of rows) {
      html += `<tr><td>${row.algorithm || "—"}</td>` + cols.map((c) => `<td>${fmt(row[c])}</td>`).join("") + "</tr>";
    }
    resultsTable.innerHTML = html;
    resultsHint.textContent = `${rows.length} row(s) loaded.`;
  }

  function renderCompare(rows) {
    compareHint.textContent = "Averaging over seeds.";
    const labels = rows.map((r) => r.algorithm);
    const datasets = [];
    const metricKeys = [
      "deadline_violation_rate_pct",
      "average_latency_s",
      "p95_latency_s",
      "total_energy_J",
      "throughput_tasks_per_s",
    ];

    metricKeys.forEach((key) => {
      const vals = rows.map((r) => (r[key] === undefined ? null : r[key]));
      const label = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
      datasets.push({
        label,
        data: vals,
        borderColor: "rgba(37, 99, 235, 1)",
        backgroundColor: "rgba(37, 99, 235, 0.15)",
        pointBackgroundColor: "rgba(37, 99, 235, 1)",
        pointRadius: 4,
      });
    });

    if (chart) {
      chart.data.labels = labels;
      chart.data.datasets = datasets;
      chart.update();
    } else {
      chart = new Chart(chartCanvas, {
        type: "bar",
        data: {
          labels,
          datasets,
        },
        options: {
          responsive: true,
          plugins: {
            legend: { position: "top" },
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${fmt(ctx.parsed.y, 4)}` } },
          },
          scales: { y: { beginAtZero: true } },
        },
      });
    }
  }

  async function runExperiment(values) {
    const res = await postJson("/run", values);
    if (!res.ok) {
      showError(runStatus, res.error || "Run failed to start.");
      return;
    }
    setControlsDisabled(true);
    showHint(runStatus, "Run started. Polling for results...");
    pollForResult();
  }

  async function pollForResult() {
    for (let i = 0; i < 60; i += 1) {
      if (runState.done) break;
      await sleep(1000);
    }
    // Timeout guard: 60 s of polling = 1 min.  If still not done, surface
    // an error instead of leaving the user waiting silently.
    if (!runState.done) {
      showError(runStatus, "Run timed out after 5 minutes — try a smaller Horizon (e.g. 300)."));
      setControlsDisabled(false);
      return;
    }
    const status = await getJson("/status");
    if (status.error) {
      showError(runStatus, status.error);
      setControlsDisabled(false);
      return;
    }
    if (!status.done) {
      showError(runStatus, "Run did not finish in time.");
      setControlsDisabled(false);
      return;
    }
    // Fetch persisted summary CSVs and render them (may be empty if not written).
    try {
      const results = await getJson("/results");
      if (results.ok && results.rows.length) {
        renderTable(results.rows);
        resultsCard.hidden = false;
        resultsHint.textContent = `Loaded ${results.rows.length} row(s) from results summary.`;
      } else {
        showError(resultsHint, "Run finished, but no results file was found.");
      }
    } catch (err) {
      showError(resultsHint, "Run finished, but results could not be read.");
    }
    try {
      const compare = await getJson("/compare");
      if (compare.ok && compare.rows.length) {
        renderCompare(compare.rows);
        compareCard.hidden = false;
        compareHint.textContent = "Mean over seeds, error bars = ±std.";
      } else {
        showError(compareHint, "No comparison rows available.");
      }
    } catch (err) {
      showError(compareHint, "Comparison could not be read.");
    }
    setControlsDisabled(false);
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const values = {
      algorithm: form.algorithm.value,
      scenario: form.scenario.value,
      n_tasks: Number(form.n_tasks.value),
      seed: Number(form.seed.value),
      horizon: Number(form.horizon.value),
      base_algorithm: form.base_algorithm.value,
    };
    runExperiment(values);
  });
})();
