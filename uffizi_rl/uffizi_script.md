# The Art of Queueing: spoken script

One spoken beat per presented slide, roughly 85 words each (about a minute), written
to be told, not read off. The seventeen main slides run the story; the appendix at the
end is Q&A backup, so those carry short "if asked" pointers rather than full beats.

---

## 1. Title

Good morning. We are Marco, Michael and Mircea. Our talk is called The Art of Queueing.
It is about the Uffizi in Florence, one of the most crowded museums on earth. The
question we ask is simple: can reinforcement learning help five thousand visitors a day
enjoy it more? Over the next fifteen minutes we take you from a tiny toy problem all the
way to a redesigned museum. Let us begin with the building itself.

---

## 2. The few rooms everyone queues for

Here is the Uffizi: ninety-eight rooms spread over two floors. The strange thing is that
almost all five thousand daily visitors come for the same handful of paintings, the
Botticellis, the Leonardos, a Raphael, a Michelangelo, crammed into just three rooms. So
those three run wildly over capacity while the other ninety-five stay calm. And the crowd
is not uniform: roughly sixty percent are Instagram tourists chasing a selfie, thirty
percent are normal visitors, ten percent are true art lovers. That mix matters later.

---

## 3. The problem and the goal

So what is wrong with this picture? Everything funnels into three rooms, which means long
queues and a crush at the masterpieces, while the rest of the museum sits half empty.
Nobody coordinates: every visitor who pushes into a packed Botticelli makes it a little
worse for the next. Economists call that a congestion externality. Our question is whether
reinforcement learning can do better. We attack it from two sides. The museum side:
redesign the incentives so the crowd spreads out. The visitor side: find the smartest way
through.

---

## 4. Two streams, one project

We split the work into two streams that stay separate until the end. Stream one is a crowd
simulator: the full sixty-thirty-ten population moving through the museum for one day. We
compare the museum as it is against a redesign, adopting it only if welfare rises and
revenue holds. It passes: plus thirty-one percent welfare, revenue preserved. Stream two,
our real focus, sits on top. We take that museum as fixed and ask what an optimal visitor
would do: when to book, which masterpieces, how to pace the day.

---

## 5. Foundations: a tabular proof of concept

Before touching the real museum, we proved the idea on a twelve-room toy small enough to
solve exactly. Crowds wave up and down without warning. Tabular Q-learning discovers
something we call temporal arbitrage: do not fight the peak, slip in during the lull just
around it. The bars on the right are telling. Learning beats every hand-written rule; the
rules even go negative when they mistime the crowd. One more check: Q-learning and
Double-Q tie almost exactly, because a deterministic toy has no overestimation bias to
remove. Theory confirmed.

---

## 6. Scaling up: deep RL and action masking

The real museum breaks that table: ninety-eight rooms, around a hundred moves at every
step, hundreds of state numbers. No table can hold that, so we switch to a neural network,
MaskablePPO, with PPO and DQN as controls. The idea is action masking. From any room only
a few moves are legal, yet the network has about a hundred outputs. So we set every illegal
move to minus infinity before the softmax, which gives it zero probability. No effort is
wasted on moves that do not exist.

---

## 7. The reframe: when to book, not where to walk

Here is the most important decision we made. A visitor has a map and a guidebook, so
walking the museum is a shortest path, already solved. The hard decision is the booking:
when to reserve each masterpiece. Booking is hard in three ways walking never
is. It is uncertain, you cannot see how fast the good slots fill. It is irreversible, book
late and the masterpiece is gone. And it is delayed, you act today but the payoff only
arrives at check-in. So we reframed the problem around booking.

---

## 8. Result: book early when busier and it pays

And here is the payoff. Both kinds of visitor beat their matched no-booking baseline at
every crowd level: the art lover gains twenty to thirty-nine percent, the tourist a steady
thirty-six to thirty-eight. All three masterpieces are secured every time. But look at the
curve on the right. The lead time the agent chooses rises with the crowd: the busier the
museum, the earlier it books. We never coded that rule. It emerged on its own from the
reward, because booking late into a packed museum simply misses the rooms.

---

## 9. What a visit looks like, on the real floor plan

It helps to see a visit. This is the art lover's learned day, eight runs drawn on the real
Uffizi floor plan, one panel per crowd level. Notice it is a single sensible route. The
agent enters at room A1, sweeps the second-floor masterpieces, the gold stars, each inside
its booked window, then takes the staircase down to Caravaggio on the first floor and
leaves before closing. And every step is a one-hop move to a neighbouring room. There are
no teleports, just a real walk through a real building.

---

## 10. Which algorithm wins and why masking matters

So which algorithm actually wins? MaskablePPO is best or tied in every single cell, stable
across seeds. Masking earns its keep exactly where the problem is hardest. At the packed
tourist, unmasked PPO collapses to 2093, below the no-booking baseline of 2680, while
MaskablePPO holds 3685. At easy crowds the two agree; the gap only opens where precision
matters. DQN, meanwhile, is fragile and noisy as the crowd grows: taking a max over noisy
crowd values overestimates, the very bias our deterministic toy did not have.

---

## 11. Challenge 1: an unlearnable reward

Those results hid pain; the struggle was the project. Here are the four hardest bugs.
First, an unlearnable reward. Our earliest agent never left; it squatted in one room
until closing. The fix we tried first, zeroing all reward when it failed to exit, made
things worse: an all-or-nothing wipe is flat everywhere, so the gradient is zero and
nothing points to the door. What worked: a dense egress signal, pressure rising near
closing and far from an exit, plus a bonus for leaving. Sparse rewards are correct but
often unlearnable.

---

## 12. Challenge 2: it could not see what it was deciding

Second bug: the agent could not see what it was deciding. The rule we wanted, stay until
satisfied then move on, never stabilised; dwell times wobbled at random. The reason was
subtle. The right move depended on how much value the visitor had already absorbed from
the room, a quantity nowhere in the state. A hidden variable. We were quietly solving a
POMDP, not the clean MDP we assumed. The fix: add a per-room appreciation-progress number
to the state, so seen enough becomes something the policy can read.

---

## 13. Challenge 3: reward shaping that backfired

Third, reward shaping that backfired. Our well-meant penalties had side effects: a boredom
penalty made the agent loop in circles; a harsh crowd penalty produced a fake art lover who
skipped the masterpieces. Every term is a new objective to game, reward hacking. Looping
beat exiting; a crowded Botticelli scored worse than skipping it. So we went minimal: a
gentle crowd discount so a crowded masterpiece still beats skipping, no boredom penalty,
satiation alone to stop lingering. Always ask: what is the cheapest way to game it?

---

## 14. Challenge 4: the do-nothing trap and the book-early trick

Fourth, our favourite. Give it a free decline button and both profiles declined all three
masterpieces; book-early stayed only half learned. The trap is that declining is safe now,
while booking only pays off later, hundreds of steps away. A classic do-nothing optimum and
a credit-assignment nightmare. Our fix had two parts. We made always books a type trait, no
escape hatch, then added an immediate off-pace penalty standing in for the distant cost. A
small signal now, tracking a far-off reward, made the long lesson learnable. Our best trick.

---

## 15. What five seeds revealed

One more habit: we never trust a single run. A single run can be lucky or unlucky, so one
number can mislead. We retrained every cell under five seeds and report the mean with the
spread. The tourist is rock-solid, near-zero variance everywhere. The art lover degrades as
the museum fills, from plus thirty-nine to plus twenty-nine to plus twenty percent. And the
catch: at maximum crowd it is bimodal. Four of five seeds reach plus forty-one, but one
collapses below baseline. We report that spread, not the lucky seed.

---

## 16. The crowd before and after the redesign

Now back to stream one and the big picture. This heatmap is the whole museum across the
whole day. The top panel is the Uffizi as it is today; the bottom is the same museum with
all eleven interventions, drawn on the same colour scale. Watch the masterpiece bands,
rooms A9 and A36: they cool from red toward orange as the worst crush drains away; the day
stretches over more hours instead of spiking at noon. The result: plus thirty-one percent
welfare, with revenue held flat. Nobody turned away.

---

## 17. Thank you

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
method and results (5 to 10), one takes the challenges and close (11 to 17). Whoever fields
a question drives to the matching backup slide above.*
