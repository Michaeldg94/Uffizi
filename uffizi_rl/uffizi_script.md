# The Art of Queueing: spoken script

One spoken beat per presented slide, roughly 85 words each (about a minute), written
to be told, not read off. The seventeen main slides run the story; the appendix at the
end is Q&A backup, so those carry short "if asked" pointers rather than full beats.

---

## 1. Title

Good morning, everyone. We're Marco, Michael and Mircea, and our talk is called The Art
of Queueing. It's about the Uffizi, in Florence — a museum many people dream of visiting,
and also one of the most crowded in the world. About five thousand people go through it
every single day. So we asked ourselves a simple question: can reinforcement learning help
these people enjoy it more? In the next fifteen minutes, we'll take you from a small toy
problem all the way to a redesigned museum. Let's start with the building itself.

---

## 2. The few rooms everyone queues for

This is the Uffizi: ninety-eight rooms, on two floors. Take a look at the map. Almost
everything people come to see is upstairs, on the second floor — the two Botticellis, the
two Leonardos, a Raphael, a Michelangelo — and all of it sits in just three rooms. The
first floor, downstairs, is much quieter. That's where Caravaggio is, and people move
around freely there. So the picture is very unbalanced: three rooms are completely full,
while the other ninety-five stay almost empty. And the crowd is not one single type. About
sixty percent are Instagram tourists, there for a selfie; thirty percent are normal
visitors; ten percent are real art lovers. Please remember that mix — it matters later.

---

## 3. The problem and the goal

So what is the problem here? Everything goes into the same three rooms. That means long
queues, and the masterpiece rooms get completely packed, while all the other rooms stay
almost empty. And nobody is coordinating. Every person who pushes into a full Botticelli
room makes it a little worse for the next one. Economists have a name for this: a
congestion externality. Our question was whether reinforcement learning could do better.
We looked at it from two sides — the museum, and the visitor. And that split is really the
whole shape of our project.

---

## 4. Two streams, one project

So those two sides become two streams of work. The easiest way to tell them apart is one
question: who are we optimising for? Stream one is the museum's problem — we change the
museum for everyone. We send the whole sixty-thirty-ten crowd through a full day, and
compare the museum as it is today with a redesign that makes the crowd spread out: booked
masterpiece slots, longer opening hours, a small extra charge at busy times. We keep the
redesign only if welfare goes up and revenue stays safe. And it does: plus thirty-one
percent welfare, revenue preserved. Stream two is the one we really care about — the
visitor's problem. Here we keep the museum fixed and look for the single best visitor
inside it: when to book, which masterpieces to see, how to plan the day.

---

## 5. Foundations: a tabular proof of concept

Before we tried this on the real museum, we first tested the idea on a small twelve-room
toy — simple enough to solve exactly. In the toy, the crowd goes up and down with no
warning. Q-learning finds a smart move we call temporal arbitrage: don't fight the peak, go
in during the quiet time just around it. Now look at the bars. The two on the left are the
learners, both close to eighty-three. Every hand-written rule is far below, and four are
even negative. Here's the surprising part: even random — just choosing by chance, at plus
thirty-five — beats all four "clever" rules. The worst rule always goes straight to the
most valuable room, no matter the time, so it reaches Botticelli right at the peak:
completely wrong about timing. Random at least catches a few quiet moments. So the lesson
is simple: when you visit matters more than whether you visit — and only learning finds
this. Last check: Q-learning and Double-Q come out equal, because in a fully predictable
toy there's no overestimation for Double-Q to fix. Exactly what the theory says.

---

## 6. Scaling up: deep RL and action masking

The real museum breaks that table: ninety-eight rooms, around a hundred moves at every
step, hundreds of state numbers. No table can hold that, so we switch to a neural network,
MaskablePPO, with PPO and DQN as controls. The idea is action masking. From any room only
a few moves are legal, yet the network has about a hundred outputs. So we set every illegal
move to minus infinity before the softmax, which gives it zero probability. No effort is
wasted on moves that do not exist.

---

## 7. As-is versus intervened: why it becomes a booking problem

We leave the toy and put the agent on the full museum, running the same algorithms in two
worlds. World one is the museum as-is: no reservations, you walk in, taking masterpieces as
you find them, so with a map the visit is a solved shortest path. World two is the
intervened museum, our eleven Pigovian measures. The headline one, RAMA, ends walk-in
access: every masterpiece needs a slot booked days ahead. That rule flips the decision from
where to walk to when to book. Booking is hard: uncertain, irreversible, delayed.

---

## 8. The eleven interventions: pricing the externality

So what does intervened mean? RAMA is one of eleven Pigovian measures in the redesign.
Remember the congestion externality from earlier: the fix is to make each visitor feel the
cost they impose and reward those who spread out. Some shift demand in time, like extended
hours and quiet-hour discounts. Some shift it in space, like reservations and enriching the
secondary rooms. Some use price, like peak pricing and a group surcharge. A director's gate
decides adoption: only if welfare rises and revenue holds. It does, plus thirty-one percent.

---

## 9. Result: book early when busier and it pays

And here is the payoff. Under RAMA, both profiles beat their matched as-is, no-booking
baseline at every crowd level: the art lover gains twenty to thirty-nine percent, the tourist
a steady thirty-six to thirty-eight. All three masterpieces are secured every time. But look
at the curve on the right. The lead time the agent chooses rises with the crowd: the busier
the museum, the earlier it books. We never coded that rule. It emerged from the reward,
because booking late into a packed museum simply misses the rooms.

---

## 10. What a visit looks like, on the real floor plan

It helps to see a visit. This is the art lover's learned day, eight runs drawn on the real
Uffizi floor plan, one panel per crowd level. Notice it is a single sensible route. The
agent enters at room A1, sweeps the second-floor masterpieces, the gold stars, each inside
its booked window, then takes the staircase down to Caravaggio on the first floor and
leaves before closing. And every step is a one-hop move to a neighbouring room. There are
no teleports, just a real walk through a real building.

---

## 11. Which algorithm wins and why masking matters

So which algorithm actually wins? MaskablePPO is best or tied in every single cell, stable
across seeds. Masking earns its keep exactly where the problem is hardest. At the packed
tourist, unmasked PPO collapses to 2093, below the no-booking baseline of 2680, while
MaskablePPO holds 3685. At easy crowds the two agree; the gap only opens where precision
matters. DQN, meanwhile, is fragile and noisy as the crowd grows: taking a max over noisy
crowd values overestimates, the very bias our deterministic toy did not have.

---

## 12. Challenge 1: an unlearnable reward

Those results hid pain; the struggle was the project. Here are the four hardest bugs.
First, an unlearnable reward. Our earliest agent never left; it squatted in one room
until closing. The fix we tried first, zeroing all reward when it failed to exit, made
things worse: an all-or-nothing wipe is flat everywhere, so the gradient is zero and
nothing points to the door. What worked: a dense egress signal, pressure rising near
closing and far from an exit, plus a bonus for leaving. Sparse rewards are correct but
often unlearnable.

---

## 13. Challenge 2: it could not see what it was deciding

Second bug: the agent could not see what it was deciding. The rule we wanted, stay until
satisfied then move on, never stabilised; dwell times wobbled at random. The reason was
subtle. The right move depended on how much value the visitor had already absorbed from
the room, a quantity nowhere in the state. A hidden variable. We were quietly solving a
POMDP, not the clean MDP we assumed. The fix: add a per-room appreciation-progress number
to the state, so seen enough becomes something the policy can read.

---

## 14. Challenge 3: reward shaping that backfired

Third, reward shaping that backfired. Our well-meant penalties had side effects: a boredom
penalty made the agent loop in circles; a harsh crowd penalty produced a fake art lover who
skipped the masterpieces. Every term is a new objective to game, reward hacking. Looping
beat exiting; a crowded Botticelli scored worse than skipping it. So we went minimal: a
gentle crowd discount so a crowded masterpiece still beats skipping, no boredom penalty,
satiation alone to stop lingering. Always ask: what is the cheapest way to game it?

---

## 15. Challenge 4: the do-nothing trap and the book-early trick

Fourth, our favourite. Give it a free decline button and both profiles declined all three
masterpieces; book-early stayed only half learned. The trap is that declining is safe now,
while booking only pays off later, hundreds of steps away. A classic do-nothing optimum and
a credit-assignment nightmare. Our fix had two parts. We made always books a type trait, no
escape hatch, then added an immediate off-pace penalty standing in for the distant cost. A
small signal now, tracking a far-off reward, made the long lesson learnable. Our best trick.

---

## 16. What five seeds revealed

One more habit: we never trust a single run. A single run can be lucky or unlucky, so one
number can mislead. We retrained every cell under five seeds and report the mean with the
spread. The tourist is rock-solid, near-zero variance everywhere. The art lover degrades as
the museum fills, from plus thirty-nine to plus twenty-nine to plus twenty percent. And the
catch: at maximum crowd it is bimodal. Four of five seeds reach plus forty-one, but one
collapses below baseline. We report that spread, not the lucky seed.

---

## 17. The crowd before and after the redesign

Now back to stream one and the big picture. This heatmap is the whole museum across the
whole day. The top panel is the Uffizi as it is today; the bottom is the same museum with
all eleven interventions, drawn on the same colour scale. Watch the masterpiece bands,
rooms A9 and A36: they cool from red toward orange as the worst crush drains away; the day
stretches over more hours instead of spiking at noon. The result: plus thirty-one percent
welfare, with revenue held flat. Nobody turned away.

---

## 18. Thank you

That is our story: from a twelve-room toy, through four hard bugs, to a single optimal
visitor and a redesigned museum that serves the same crowd better. Thank you. We are happy
to take your questions; we have a stack of backup slides ready for the details.

---

# Appendix: backup slides (use if asked)

These are not narrated in sequence. Each is a one-line answer to a likely question.

- **A1. The three MDPs.** Same five MDP elements in the toy, the as-is walk and RAMA; only
  the problem, the state encoding and the action screening change.
- **A2. The as-is state, eight numbers.** What the visitor sees on the planned walk, in
  natural units, with the example vector at the door A1.
- **A3. The RAMA state, twenty-five numbers.** The booking-task state, grouped; the booking
  screen is the crux.
- **A4. Actions and masking.** The action space per setting and the masked-softmax formula.
- **A5. The reward functions.** Exact terms for the toy, the full museum, plus the
  booking-only off-pace and no-show penalties.
- **A6. Booking phase and fill curve.** The two-phase episode and the busy-day fill curve:
  one slot free a day before, eight a month ahead.
- **A7. Full booking results.** Every number behind the grid, five seeds, mean and std;
  note the bimodal art lover at maximum crowd.
- **A8. Algorithm detail.** The same algorithm chart with the three takeaways spelled out.
- **A9 / A10. The learned walks.** Baseline versus RAMA on the floor plan, art lover then
  tourist.
- **A11. Per-profile density.** Crowd heatmaps by profile, baseline versus RAMA.
- **A12. Interventions and the gate.** The welfare-revenue gate and the eleven-measure
  portfolio.
- **A12b. Ranking the interventions.** Candidate measures ranked by standalone welfare gain;
  extended hours and enrichment lead, the annual pass goes negative.
- **A13. Robustness sweeps.** The welfare gain holds as we vary Instagram share, volume and
  heterogeneity.
- **A14. Toy internals.** The learning curve and the Q versus Double-Q tie.
- **A15. Training and evaluation.** Hyperparameters, the five seeds and the
  common-random-number evaluation.
- **A16. Limitations and next steps.** One visitor, not an equilibrium; fixes we did not run
  (Double-DQN, a tougher fixed-heuristic baseline, an off-pace ablation); what is baked in.

---

*Suggested three-way split: one presenter takes the setup (slides 1 to 4), one takes the
method and results (5 to 11), one takes the challenges and close (12 to 18). Whoever fields
a question drives to the matching backup slide above.*
