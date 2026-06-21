# The Art of Queueing: spoken script

One spoken beat per presented slide, roughly 85 words each (about a minute), written
to be said out loud, not read off. The seventeen main slides run the story; the appendix
at the end is Q&A backup, so those carry short "if asked" pointers rather than full beats.

---

## 1. Title

Hi everyone. We're Marco, Michael and Mircea, and our project is called The Art of
Queueing. It's about the Uffizi, the big art museum in Florence. It gets really crowded, and
we wanted to find out: can reinforcement learning actually help with that? So over the next
fifteen minutes, we'll start small, with a simple toy problem, and build all the way up to a
full museum, and a visitor who knows exactly how to move through it. But first, let me show
you the place.

---

## 2. The Uffizi Gallery

The Uffizi is one of those places that's on everyone's list. And you can see why. These are
some of the most famous paintings in the world. But here's the catch. Almost all of them hang
on the same floor, upstairs. Downstairs, where Caravaggio is, it's actually pretty calm. So you
have thousands of people arriving every single day, and nearly all of them are trying to reach
the same few rooms. And that's really where our whole problem begins.

---

## 3. The problem: where RL comes in

Let me walk through this chart, it's the heart of the problem. Each bar is a room, and its
length shows how packed it gets at the busiest moment, compared to what it's built to hold. The
dashed line is the limit: a room exactly full. Anything past it is over capacity. Look, the top
three, Botticelli, Leonardo and Raphael, almost hit double. Those are the red ones. Every other
room sits well to the left, with space to spare. So the museum isn't too small. Everyone is just
fighting for the same three rooms. And that's what we wanted RL to fix.

---

## 4. The plan

Our plan is a loop. First, we build a digital copy of the museum, where every room has a size,
exits, and a value. Then we add a realistic crowd, and we track two things: how much money the
museum makes, and how happy the visitors are. Once it matches the real Uffizi, we drop a
learning agent in to find the smartest visit. Then we change things like the prices and the
hours, keeping only what makes visitors happier without losing money. And we re-run it to see
what improved.

---

## 5. A toy first

Before the real museum, we started tiny. Just twelve rooms, simple enough to solve by hand. In
it, rooms fill up and empty out through the day, and the agent learns the trick by itself: go to
the busy rooms when they're quiet. In this chart, each bar is the score for a whole visit, so
higher is better. The two on the left are our learning methods, Q-learning and Double-Q. The
other five are simple rules we wrote by hand, like just following the standard tourist route.
Learning beats all five, and some of them even do worse than nothing. The two learners tie,
exactly like the theory predicts.

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
