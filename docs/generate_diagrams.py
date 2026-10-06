import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DOCS_DIR, exist_ok=True)

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0a0f1d")
    ax.set_facecolor("#0a0f1d")
    ax.axis("off")

    # Title
    ax.text(7, 9.6, "Reliable Credit Risk MLOps Architecture", fontsize=20, fontweight="bold", color="#38bdf8", ha="center")
    ax.text(7, 9.2, "Automated Failure Detection, Quality Gates, Rollback & Self-Healing Pipeline", fontsize=12, color="#94a3b8", ha="center")

    boxes = [
        {"id": "git", "label": "GitHub Repository\n& CI/CD Actions", "pos": (1.5, 7.8), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#38bdf8"},
        {"id": "airflow", "label": "Apache Airflow\nOrchestration DAG", "pos": (5.5, 7.8), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#a855f7"},
        {"id": "validation", "label": "Data Validation\n& Preprocessing", "pos": (9.5, 7.8), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#10b981"},
        
        {"id": "train", "label": "3-Model Training\n(LR, RF, XGBoost)", "pos": (9.5, 6.0), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#38bdf8"},
        {"id": "mlflow", "label": "MLflow Tracking\n& Registry", "pos": (5.5, 6.0), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#f59e0b"},
        {"id": "quality", "label": "Quality Gate Shield\n(F1 & ROC-AUC Check)", "pos": (1.5, 6.0), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#ef4444"},

        {"id": "champion", "label": "Champion / Challenger\nModel Manager", "pos": (1.5, 4.2), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#38bdf8"},
        {"id": "fastapi", "label": "FastAPI Serving\n(REST API & Web UI)", "pos": (5.5, 4.2), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#10b981"},
        {"id": "monitoring", "label": "Prometheus & Grafana\nTelemetry Monitoring", "pos": (9.5, 4.2), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#a855f7"},

        {"id": "drift", "label": "Evidently & Statistical\nDrift Detection", "pos": (9.5, 2.2), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#f59e0b"},
        {"id": "diagnosis", "label": "Automated Diagnosis\n& Alert Engine", "pos": (5.5, 2.2), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#ef4444"},
        {"id": "rollback", "label": "Rollback Engine\n& Backup Champion", "pos": (1.5, 2.2), "w": 3.0, "h": 0.9, "color": "#1e293b", "border": "#10b981"},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch(
            b["pos"], b["w"], b["h"],
            boxstyle="round,pad=0.1",
            linewidth=2,
            edgecolor=b["border"],
            facecolor=b["color"]
        )
        ax.add_patch(rect)
        cx = b["pos"][0] + b["w"] / 2
        cy = b["pos"][1] + b["h"] / 2
        ax.text(cx, cy, b["label"], color="#f8fafc", fontsize=10, fontweight="bold", ha="center", va="center")

    # Connecting Arrows
    arrow_props = dict(arrowstyle="->", color="#94a3b8", lw=2, mutation_scale=15)
    
    # Row 1
    ax.annotate("", xy=(5.4, 8.25), xytext=(4.6, 8.25), arrowprops=arrow_props)
    ax.annotate("", xy=(9.4, 8.25), xytext=(8.6, 8.25), arrowprops=arrow_props)
    
    # Row 1 to Row 2
    ax.annotate("", xy=(11.0, 7.0), xytext=(11.0, 7.7), arrowprops=arrow_props)
    
    # Row 2
    ax.annotate("", xy=(8.6, 6.45), xytext=(9.4, 6.45), arrowprops=arrow_props)
    ax.annotate("", xy=(4.6, 6.45), xytext=(5.4, 6.45), arrowprops=arrow_props)

    # Row 2 to Row 3
    ax.annotate("", xy=(3.0, 5.2), xytext=(3.0, 5.9), arrowprops=arrow_props)
    
    # Row 3
    ax.annotate("", xy=(5.4, 4.65), xytext=(4.6, 4.65), arrowprops=arrow_props)
    ax.annotate("", xy=(9.4, 4.65), xytext=(8.6, 4.65), arrowprops=arrow_props)

    # Row 3 to Row 4
    ax.annotate("", xy=(11.0, 3.2), xytext=(11.0, 4.1), arrowprops=arrow_props)

    # Row 4 (Recovery loop)
    ax.annotate("", xy=(8.6, 2.65), xytext=(9.4, 2.65), arrowprops=arrow_props)
    ax.annotate("", xy=(4.6, 2.65), xytext=(5.4, 2.65), arrowprops=arrow_props)
    ax.annotate("", xy=(3.0, 4.1), xytext=(3.0, 3.2), arrowprops=dict(arrowstyle="->", color="#10b981", lw=2, mutation_scale=15))

    ax.set_xlim(0, 14)
    ax.set_ylim(1, 10)
    
    out_path = os.path.join(DOCS_DIR, "architecture.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0a0f1d")
    plt.close()
    print(f"Architecture diagram generated: {out_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
