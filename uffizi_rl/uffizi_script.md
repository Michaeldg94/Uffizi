# The Art of Queueing: spoken script

One spoken beat per presented slide, roughly 85 words each (about a minute), written
to be said out loud, not read off. The seventeen main slides run the story; the appendix
at the end is Q&A backup, so those carry short "if asked" pointers rather than full beats.

---

## 1. Title

Good morning. We're Marco, Michael and Mircea. Our talk is The Art of Queueing. It's about
the Uffizi in Florence, one of the most crowded museums anywhere; the question is whether
reinforcement learning can help. Over the next fifteen minutes we go from a tiny toy problem
to a redesigned museum and the optimal visitor inside it. Let's start with the building.

---

## 2. The Uffizi Gallery

This is the Uffizi: about five thousand visitors a day through ninety-eight rooms on two
floors. These are the paintings nearly all of them come for, the two Botticellis, the
Leonardos, a Raphael, a Michelangelo, all upstairs on the second floor. Downstairs is
quieter, with Caravaggio. Keep that layout in mind, because the whole problem comes from
where everyone wants to be.

---

## 3. The problem: where RL comes in

Here's the problem; the figure makes it brutal. At the daily peak, three rooms, the
Botticelli, Leonardo and Raphael rooms, run at roughly twice their capacity. The red bars are
past the line; everything else sits well below. So a few rooms are jammed while most of the
museum is nearly empty. Our question is whether reinforcement learning can help us understand
this and simulate a calmer, better-run Uffizi.

---

## 4. The plan

The plan has five steps. One: rebuild the Uffizi as a virtual model, every room with its size,
doorways and a value, because rooms differ: a Botticelli counts for more than a stairwell.
Two: generate the crowd from real numbers, about sixty percent phone-first tourists, thirty
percent ordinary, ten percent art lovers; check it matches the real museum, then read off
revenue and welfare. Three: drop an RL agent on the museum as it is. Four: intervene, keeping
only changes that hold revenue while lifting welfare. Five: re-run the agents and see who
gains.

---

## 5. A toy first

Before the real museum, we test on a twelve-room toy small enough to solve exactly. The
crowds rise and fall with no warning; Q-learning learns to time them: go into a room in the
lull, not at the peak. The bars show it beats every hand-written rule, some of which actually
score negative. Q-learning and Double-Q tie, because a deterministic toy has no overestimation
bias to fix, which is what the theory says.

---

## 6. The toy does not scale

The real museum breaks that table: ninety-eight rooms, around a hundred moves at each step,
hundreds of numbers in the state. A lookup table that big could never fill, so we move to a
neural net, MaskablePPO, with PPO and DQN as controls. The key trick is masking. From any room
only a few moves are physically legal, so we set the illegal ones to minus infinity before the
softmax. The agent never wastes time learning which moves don't exist; they're simply off the
table.

---

## 7. The museum as it is: the baseline

First, the museum as it is today. We calibrate it to the real hours, prices and surcharges,
then drop our agents on it, where the visitor just walks, no booking. This is the ceiling with
no change: the art lover scores around five thousand, dropping toward four thousand as the
rooms fill; the tourist sits near twenty-seven hundred. That's today's Uffizi; every later
number is measured against it.

---

## 8. The interventions and the museum-wide payoff

Now we redesign. The intervened museum bundles eleven measures, each one charging for the
crowding a visitor causes: longer hours and quiet-hour discounts on time, reservations and
richer side rooms on access, peak pricing and a group surcharge on price. The heatmap shows
the effect: top is today, bottom is all eleven, same scale. The masterpiece bands cool from
red toward orange; the day spreads over more hours. Welfare is up thirty-one percent with
revenue held.

---

## 9. After the redesign: a booking problem

The headline intervention is RAMA: masterpieces now need a booked slot, so you can't just walk
in. That flips the visitor's real decision from where to walk to when to book, one, seven,
twenty-one or thirty-five days ahead. It's hard because it's uncertain, irreversible and
delayed. And it pays: the art lover's ceiling climbs from five thousand to seven thousand, the
tourist's from twenty-seven hundred to thirty-seven hundred, all masterpieces secured. The
agent even learns to book earlier when the museum is busier, which we never coded.

---

## 10. Two visits: the art lover roams, the tourist beelines

Here's how the two visitors actually move, drawn on the floor plan. The art lover, on top,
works the whole museum: its trace covers the second floor end to end and fans across the first
floor too. The normal tourist, on the bottom, does not. After the masterpieces it skips the B
block and rushes through C and D straight to Caravaggio, then leaves. Same museum, two
completely different visits.

---

## 11. Which algorithm wins and why masking matters

Which algorithm got us here? The pale bars are the two no-booking baselines, the coloured ones
the three booking agents. MaskablePPO, in dark blue, is best or tied everywhere and stable
across seeds; it's the winner. Unmasked PPO, in orange, collapses at the packed tourist, below
the baseline, because it wastes effort on illegal moves. Masking only matters at the hardest
cells, but there it decides the result. DQN, in green, is fragile and noisy: it overestimates
by taking a max over noisy values.

---

## 12. Challenge 1: an unlearnable reward

Now the hard parts; the struggle was most of the project. First, an unlearnable reward. Our
early agent never left; it sat in one room until closing. Zeroing the reward when it failed to
exit made things worse: the wipe is flat everywhere, so the gradient is zero and nothing
points to the exit. What worked was a dense signal, pressure rising near
closing and far from a door, plus a bonus for leaving. Even a correct reward is useless if
it's flat; gradient descent needs a slope to follow.

---

## 13. Challenge 2: it could not see what it was deciding

Second, the agent couldn't see what it was deciding. The rule stay until satisfied then move
never settled; dwell times wobbled at random. The reason: the right move depended on how much
the visitor had already gotten out of the room, which wasn't in the state. We'd quietly turned
it into a POMDP. The fix was a per-room appreciation-progress number, so seen enough is
something the policy can read. Whatever the best move depends on has to be in the state.

---

## 14. Challenge 3: reward shaping that backfired

Third, our reward shaping backfired. A boredom penalty made the agent loop in circles; a harsh
crowd penalty produced a fake art lover that skipped the masterpieces. Every term you add is a
new thing to game: looping beat exiting, a crowded Botticelli scored worse than just skipping
it. So we went minimal, a gentle crowd discount so a packed masterpiece still beats skipping,
no boredom penalty, satiation alone to stop lingering. The agent games every term the laziest
way it can.

---

## 15. Challenge 4: the do-nothing trap and the book-early trick

Fourth, our favourite. Given a free decline button, both profiles declined all three
masterpieces; book-early only half learned. Declining is safe right now, while booking only
pays much later, hundreds of steps away, a classic do-nothing trap. The fix had two parts:
make always-books a fixed trait with no escape, then add an immediate off-pace penalty
standing in for that distant cost. A small penalty now, standing in for a cost far in the
future, was what finally got it booking. Nothing else helped as much.

---

## 16. Conclusion

To wrap up. We rebuilt the Uffizi, matched it to the real museum, then used it to ask two
things: how to redesign the building and how a single visitor should play it. The redesign,
eleven measures, lifts welfare thirty-one percent with revenue held. The optimal visitor,
MaskablePPO, books early and beats its old ceiling at every crowd level. And the lesson:
choosing what the agent decides was far harder than the algorithm itself; masking did more for
us than any tuning.

---

## 17. Thank you

That's the story: from a twelve-room toy, through four hard bugs, to a redesigned museum and
an optimal visitor inside it. Thank you. We're happy to take questions; there's a stack of
backup slides if you want the details.

---

# Appendix: backup slides (use if asked)

These are not narrated in sequence. Each is a one-line answer to a likely question.

- **A1. The three MDPs.** Same five MDP elements in the toy, the as-is walk and RAMA; only the
  problem, the state encoding and the action screening change.
- **A2. The as-is state, eight numbers.** What the visitor sees on the planned walk, in natural
  units, with the example vector at the door A1.
- **A3. The RAMA state, twenty-five numbers.** The booking-task state, grouped; the booking
  screen is the crux.
- **A4. Actions and masking.** The action space per setting and the masked-softmax formula.
- **A5. The reward functions.** Exact terms for the toy, the full museum, plus the booking-only
  off-pace and no-show penalties.
- **A6. Booking phase and fill curve.** The two-phase episode and the busy-day fill curve: one
  slot free a day before, eight a month ahead.
- **A7. Full booking results.** Every number behind the grid, five seeds, mean and std; note
  the bimodal art lover at maximum crowd.
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
method and results (5 to 11), one takes the challenges and close (12 to 17). Whoever fields a
question drives to the matching backup slide above.*
