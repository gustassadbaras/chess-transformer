# 2026-09-27
* Just found out what I've been referring to as "moves" are actually "half-moves" or "plies", though not planning on changing it at this point, just need to write this down somewhere;
* I'm contemplating if I should even continue running/benchmarking the 3-token-per-move models as:
  *  It seems they should require 9x as many computations as 1-token-per-move models (sequences 3x as long & each move requires 3 inferences), which should be at least 3x slower assuming perfect parallelism?
  *  Datasets use substantially more RAM, take longer to initialize;
  *  the code they requires is more complicated, certain parts of eval are much more difficult / drastically more inefficient;
  *  Regarding the promotion token of each move, these are mostly <|SKIP|> within the training data. This makes validation and training loss values more difficult to interpret. And of course, class imbalance - surely very bad? 
  *  The only significant benefit I can think of relates to weight sharing / good inductive bias considering, for example, that moves made from specific squares or to specific squares are mathematically related, by construction (they share 1/3 of their representative tokens). Might be good for pattern recognition, also possibly better data efficiency. However, I have more than enough data and I can think of other tokenizers/representations which would likely also have these benefits with far fewer drawbacks; 
  *  It's awkward to improve upon. For example, board-state models (64 input tokens, one for each square) would still require three separate inferences for a full move (half move), which implies the already-generated 1-or-2 tokens must be fed into the model somehow. Could just append them to the 64 tokens, but that's awkward;
Though to be fair, I'm now intrigued by how it would perform in comparison, given how many drawbacks I can name. Though the number of confounds would probably be ridiculous (e.g. presumably 9x as many FLOPs for the 3-token-per-move models, not even sure how I'd compensate for that);
* I think there's much reason to believe that concentrating on board-state models, and perhaps RL fine-tuning, would lead to much more progres than creating a perfect experiment to prove an architecture is bad when I can already name a dozen strong weaknesses.

I just realized and should note that it's unclear how much the "3x longer sequences" slows training/inference, and how efficiently those additional computations would be used (important if I were try to resolve confounds, perhaps?). Also, I think my use of "FLOP" may have been wrong. 

