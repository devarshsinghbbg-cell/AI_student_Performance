# EduMentor AI - user study protocol (pilot, 8-12 participants)

**Participants:** 6-8 students and 2-4 faculty/advisors, volunteers, 18+. Inform each person of the purpose, that participation is voluntary,
that no real marks are needed (they use the preset or invented data), and that they can stop at any time. Record no names; use P1, P2, ...
Check whether your department requires approval for studies with human participants before starting.

**Tasks (about 15 minutes):**
1. Student: load the "borderline student" preset, then change attendance and missed deadlines and describe how the risk changes.
2. Student: name the two main reasons given for the risk score and one action from the study plan.
3. Student: mark a topic as hard and say how the plan changed.
4. Faculty: open the cohort view, filter High risk, select a student and say which intervention you would use.
5. Both: create an account, save two snapshots and describe the trend chart.

**Measures:** (a) task success (yes/no) and time; (b) the 10-item System Usability Scale, answers 1 (strongly disagree) to 5 (strongly agree) saved as q1..q10 in
`responses.csv`; (c) three 1-5 ratings: "The reasons were easy to understand", "The study plan was useful", "I would act on this advice"; (d) one open comment.

**SUS items:** 1 I think I would like to use this system frequently. 2 I found the system unnecessarily complex. 3 I thought the system was easy to use.
4 I think I would need technical support to use it. 5 The functions were well integrated. 6 There was too much inconsistency. 7 Most people would learn it quickly.
8 I found it very cumbersome to use. 9 I felt very confident using it. 10 I needed to learn a lot before I could get going.

**Analysis:** run `python score_sus.py responses.csv`; report mean and SD of SUS (68 is the usual average benchmark), success rates, mean ratings and a short summary of comments.
With so few participants, treat results as indicative, not statistically conclusive.
