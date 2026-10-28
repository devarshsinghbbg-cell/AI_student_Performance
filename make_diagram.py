import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec="#16204a", lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.2, color="#16204a")
def arrow(x0, y0, x1, y1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color="#16204a", lw=1.2))
box(1, 17, 16, 14, "Data inputs\nattendance, quizzes,\nassignments, coding &\nstudy habits\n(CSV / form)", "#e7ecff")
box(22, 17, 15, 14, "Pre-processing\ncleaning, feature\nengineering,\nstandardisation", "#e7ecff")
box(42, 17, 15, 14, "Risk model\nLogistic Regression\n(compared with DT,\nRF, GB)", "#ffe8cc")
box(62, 28, 17, 12, "Explainability\nper-feature contribution\nto each student's score", "#e3f4e7")
box(62, 6, 17, 12, "Recommendation engine\nweak subjects, habits,\nweekly study plan", "#e3f4e7")
box(84, 28, 14, 12, "Student web app\nrisk, reasons, plan", "#fbe3e3")
box(84, 6, 14, 12, "Faculty dashboard\ncohort risk, actions", "#fbe3e3")
for a in [(17.6,24,21.4,24),(37.6,24,41.4,24),(57.6,26,61.4,33),(57.6,22,61.4,13),(79.6,34,83.4,34),(79.6,12,83.4,12),(70.5,27.6,70.5,18.6)]:
    arrow(*a)
arrow(91, 5.4, 91, 2.5, "-"); arrow(91, 2.5, 9, 2.5, "-"); arrow(9, 2.5, 9, 16.4)
ax.text(50, 4.2, "feedback loop: new quiz / assignment data re-scores the student and refreshes the plan", ha="center", fontsize=7.5, style="italic", color="#16204a")
plt.tight_layout(); plt.savefig("figures/architecture.png", dpi=200); print("ok")
