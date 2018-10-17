-module(hw1).

% functions for Q1: How long does length take?
-export([q1_fun/1]).

% functions for Q2: dot product
-export([dot_prod/2, dot_prod_hr/2, dot_prod_guard/2, dot_prod_tr/2]).

% functions for Q3: matrix multiplication
-export([transpose/1, matrix_mult/2]).

% functions for Q4: The Life of Brian
-export([brian_matrix/0, markov_step/2, markov_steps/3, markov_fixed_point/3]).

% functions for Q5: Will Brian pass?
-export([pass/2, pass/6, pass_count/3, pass_prob/3, pass_prob_par/4]).
-export([next_state/2, matrix_vector_mult/2]).

% utilities
-export([your_answer/2]). % I export this so that your code will compile
                          % without warnings when you've replaced all
			  % calls to your_answer with your answer.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q1: how long does length take?                           %
%     You need to make timing measurements for length/1 and q1_fun/1       %
%     (length is a built-in Erlang function).  You should report your      %
%     measurements in hw1.pdf.  You can make your measurements from the    %
%     Erlang shell, or add a function here for measuring and reporting     %
%     the execution times.                                                 %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% q1_fun() -- a simple function for timing measurement experiments.
%   q1_fun() -> sum_{i=0}^{length(List)} i = length(List)*(length(List)-1) div 2
q1_fun([]) -> 0;
q1_fun(List = [_ | Tl]) -> length(List) + q1_fun(Tl).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q2: dot product                                          %
%     You need to complete dot_prod_tr.                                    %
%     You need to add test cases for dot_prod_tr to hw1_tests.erl.         %
%     You need to make timing measurements for dot_prod, dot_prod_hr,      %
%     dot_prod_guard, and dot_prod_tr.                                     %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dot_prod(V1, V2) -> compute the dot product of vectors V1 and V2:
%   sum_{i=1}^n V1_i * V2_i
%   where n is the length of V1 and V2; V1_i denotes the i^th element of V1;
%   and likewise for V2_i.
%
%   Example:
%     dot_prod_tr([2, 3], [4, 5]) ->  (2 * 4) + (3 * 5) -> 23
%
%   V1 and V2 should be lists of numbers and should have the same length.
%   dot_prod(V1, V2) is a very "functional" version using zip and a list
%   comprehension.
dot_prod(V1, V2) ->
  lists:sum([X*Y || {X,Y} <- lists:zip(V1, V2)]).

% dot_prod_hr(V1, V2) -> a head recursive implementation of dot product
dot_prod_hr([X | TlX], [Y | TlY]) ->
  X*Y + dot_prod_hr(TlX, TlY);
dot_prod_hr([], []) -> 0.

% dot_prod_guard(V1, V2) -> this is based on dot_prod_hr.  This time we
%   added a guard to explicitly check that V1 and V2 are of the same
%   length.  By checking is_number of the heads of the list at each
%   call, we're enforcing that they are lists of numbers.
%   What impact does this guard checking have on the run-time?
dot_prod_guard([X | TlX], [Y | TlY]) when is_number(X), is_number(Y), length(TlX) == length(TlY) ->
  X*Y + dot_prod_hr(TlX, TlY);
dot_prod_guard([], []) -> 0.

% dot_prod_tr(V1, V2) -> a tail-recursive implementation of dot product.
%   You need to write this one.  I expect that you will use a helper
%   function, e.g. dot_product_tr(V1, V2, Acc).
%
%   Example:
%     dot_prod_tr([2, 3], [4, 5]) ->  (2 * 4) + (3 * 5) -> 23
%
%   Is the tail-recursive version faster than dot_prod or dot_prod_hr?

dot_prod_tr(V1, V2) -> dot_prod_tr(V1, V2, 0). % V1 and V2 must be lists of numbers of the same length.
dot_prod_tr([], [], Acc) -> Acc;
dot_prod_tr([H1 | Tl1], [H2 | Tl2], Acc) -> dot_prod_tr(Tl1, Tl2, Acc + H1 * H2).
	



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q3: matrix multiplication                                %
%     You need to complete matrix_mult.                                    %
%     You need to add test cases for matrix_mult to hw1_tests.erl.         %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% matrices are represented as lists of rows, where each row is a list of
%     numbers.  This means we can't have a 0-by-m matrix, but this we won't
%     worry about corner cases of various empty matrices where the number of
%     rows and/or columns is 0.
%  All rows should be the same length.

% transpose(A) -> swap row and column indices of A.
%  I provided transpose so you won't have to write it.
transpose(A = [[_|_]|_]) ->  % this pattern says that the first row is non-empty.
  % ColHd is a list consisting of the first element of each row.  This will
  %   throw an exception if any row is empty.  That enforces our requirement
  %   that all rows have the same length.
  % ColTl is a list of the remaining elements of each row.  In other workds,
  %   if A is an n-by-m matrix, then ColTl will be a n-by-(m-1) matrix, the
  %   matrix obtained by deleting the first column of A.
  {ColHd, ColTl} = lists:unzip(lists:map(fun([Hd | Tl]) -> {Hd, Tl} end, A)),
  [ColHd | transpose(ColTl)];
transpose([[] | RowTl]) -> % the first row is empty
  % make sure that all of the other rows are empty.  The function passed to
  %   the map will throw a function_clause exception if called with anything
  %   other than [].
  lists:map(fun([]) -> ok end, RowTl),
  []. % all rows are empty, return [].

%  matrix_mult(A, B) -> C
%   A, B: matrices.  The number of columns of A must equal the number of
%     rows of B.
%   C: the matrix product of A, B.
%   
%   Let
%       element(A, I, J) -> lists:nth(J, lists:nth(I, A))
%     Explanation: lists:nth(I, A) is the I^th row of A.
%       lists:nth(J, lists:nth(I, A)) is the element in the J^th column of
%         I^th row of A.
%   Let K be the number of columns of A (equal to the number of rows of B).
%   Then, C(I,J) = sum_{M=1}^K element(A, I, K) * element(B, K, J).
%
%   Example:
%     A = [[1,2], [3,4]].
%     B = [[5,6], [7,8]].
%     matrix_mult(A, B) -> [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
%                          = [[19, 22], [43, 50]].
matrix_mult(A,B) -> matrix_mult_int(A, transpose(B)).
matrix_mult_int([],_) -> [];
matrix_mult_int([Hd | Tl],B) -> 
	[mult_row_col(Hd, B) | matrix_mult_int(Tl, B)].
mult_row_col(_,[]) -> [];
mult_row_col(R,[ColHd | ColTl]) ->
	[dot_prod(R, ColHd) | mult_row_col(R, ColTl)].


% matrix_vector_mult(M, V)
%   Parameters:
%     M: a n-by-m matrix
%     V: a vector with m-elements
%   Result:
%     V2: a vector with n elements.  V2 = M*V
%   Example:
%     M = [[1,2], [3,4]].
%     V = [5,6].
%     hw1:matrix_vector_mult(M, V) -> [1*5 + 2*6, 3*5 + 4*6] = [17, 39].
%   Notes:
%     matrix_vector_mult is useful for Q4.  I'm providing it for "free" so
%     so you can focus on the more interesting aspects of the assignment.
%     We can treat V convert V to a m-by-1 matrix.
%     That means each element of V (a number) becomes a singleton list.
%     matrix_mult produces an n-by-1 matrix as a result (a list of singleton lists).
%     We convert that back to a vector (i.e. a list of numbers).
matrix_vector_mult(M, V) ->
  [X2 || [X2] <- matrix_mult(M, [[X1] || X1 <- V])].


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q4: The Life of Brian                                    %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

brian_matrix() -> % see figure "The Matrix" from the homework assignment
  [ [0.0, 0.4, 0.2, 0.4, 0.2, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0],
    [0.2, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
    [0.1, 0.0, 0.3, 0.0, 0.1, 0.4, 0.0, 0.0],
    [0.4, 0.4, 0.3, 0.6, 0.0, 0.6, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.5, 0.0] ].


% markov_step(M, Prob0) -> Prob1
%   Parameters:
%     M: a n-by-n matrix of transition probabilities.
%          Every element of M should be in the interval [0,1],
%          and the elements of each column of M should sum to 1.
%     Prob0: a vector of probabilities.  lists:nth(I, Prob0) is
%          the probability of being in state I.
%   Result:
%     Prob1: a vector of probabilities.  The probabilities for
%          each state after a step according to transition matrix M.
%   Notes:
%     We'll say that Prob0 is a valid state vector if the elements of
%     Prob0 are in the interval [0,1] and sum to 1.  In English, the
%     probability that we are in some state is exactly one.
%     Likewise for Prob1.  If M satisfies the property that the elements of
%     each column are in [0,1] and sum to 1, and Prob0 is a valid state
%     vector, then Prob1 will be a valid state vector as well.  But, we're
%     doing the computation using floating point arithmetic with round-off
%     errors.  Dividing Prob2 by the sum of its elements keeps the floating
%     point result close to the exact mathematical answer.
%
%   Example:
%     M = brian_matrix().
%     Prob0 = [0.10, 0.35, 0.20, 0.15, 0.09, 0.05, 0.03, 0.03].
%     markov_step(M, Prob0) -> [0.258, 0.097, 0.108, 0.099, 0.36, 0.024, 0.03, 0.024].
markov_step(M, Prob1) ->
  Prob2 = matrix_vector_mult(M, Prob1),
  Sum = lists:sum(Prob2),
  [ X/Sum || X <- Prob2 ].

% markov_steps(M, Prob0, N) -> ProbN
%   Parameters:
%     M, Prob0: same as for markov_step(M, Prob0)
%     N: take N steps.
%   Result:
%     ProbN: a vector of probabilities.  The probabilities for
%          each state after taking N steps starting from Prob0
%          according to transition matrix M.
%
%   Example:
%     markov_steps(hw1_sol:brian_matrix(), [0,0,0,1,0,0,0,0], 3) ->
%       [0.232, 0.124, 0.116, 0.112, 0.324, 0.0160, 0.060, 0.016]
markov_steps(_, Prob, 0) -> Prob;
markov_steps(M, Prob, N) -> markov_steps(M, markov_step(M, Prob), N-1).


% markov_fixed_point(M, Prob0, ErrTol) -> ProbInf
%   Under "reasonable" conditions, repeated multiplication by M will converge
%   to the steady-state distribution for the state.  In other words, if we
%   wait "long enough", the probabilities for what Brian is doing won't
%   depend on precisely how long we waited.  We estimate this distribution
%   by repeatedly multiplying Prob by M and returning when the difference
%   between the elements of Prob and M*Prob is less than ErrTol.
%     If you've had a class in probability theory, you know there are
%   matrices for which this simple approach won't converge.  This isn't
%   a probability theory course; so we won't go into a discussion of
%   ergodicity, or how we could perturb M to ensure that it is recurrent
%   and aperiodic.
%
%   Example:
%     markov_fixed_point(M, [0.3, 0.3, 0.4], 1.0e-10) ->
%       [0.4699, 0.2530, 0.2771].  % answer rounded by Mark
markov_fixed_point(M, Prob1, ErrTol) -> 
	Prob2 = markov_step(M, Prob1),
	MaxDiff = lists:max([abs(X-Y) || {X,Y} <- lists:zip(Prob1, Prob2)]),
	if 
		MaxDiff < ErrTol ->
			Prob2;
		MaxDiff == ErrTol ->
			Prob2;
		true ->
			markov_fixed_point(M, Prob2, ErrTol)
	end.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q5: Will Brian pass?                                     %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% next_state(M, CurrentState) -> NextState
%   For simulating a random system (i.e. a Markov chain).
%   Parameters:
%     M: the transition probability matrix.
%     CurrentState: the current state, an integer.
%   Result:
%     NextState: the next state to use in the simulation.  An integer.
%   Examples:
%     next_state(brian_matrix(),1) -> 2 with probability 30% (eat -> study)
%     next_state(brian_matrix(),1) -> 3 with probability 20% (eat -> lecture)
%     next_state(brian_matrix(),1) -> 4 with probability 10% (eat -> sleep)
%     next_state(brian_matrix(),1) -> 5 with probability 40% (eat -> snapchat)
%     For fun, you can try:
%       42> X = [hw1:next_state(hw1:brian_matrix(), 1) || _ <- lists:seq(1,1000)].
%       [5,5,5,4,2,2,3,4,3,5,5,2,5,4,5,2,3,5,3,5,2,2,5,2,5,4,2,4,5|...]
%       43> length([E || E <- X, E == 2]).  % how many 2's are in X.
%       286  % you may get a different number depending on how many times your code
%            %   has called rand:uniform before trying this example
%       % We could continue and count the number of occurrences of each other state.
%       % Let's do them all with one command.
%       43> [    length([E || E <- X, E == I])  % how many I's are in Y.
%             || I <- lists:seq(1,8) ].
%       [0,286,184,104,426,0,0,0]
%       % Looks good.  We could compute the variance for the number of occurrences
%       % of each state and then check if this is a plausible outcome for the
%       % given distribution, but this isn't a statisticis course.
%       
%   Notes:
%     The simulation produces a trajectory in accordance with the
%     probabilities of M.  We use rand:uniform() to get a pseudo-random
%     number in the interval [0,1] and use that to select the NextState
%     according to the entries in the CurrentState column of M.
next_state(M, CurrentState) ->
  Col = [ lists:nth(CurrentState, Row) || Row <- M ],
  R = rand:uniform(),
  {NextState, _} = lists:foldl(fun(P, {Index, AccProb}) when AccProb < R -> {Index+1, AccProb+P};
                                  (P, {Index, AccProb}) -> {Index, AccProb+P}
			       end, {0, 0.0}, Col),
  NextState.

% pass(M, PartyMax, SleepMax, CurrentState, PartyCount, SleepCount) -> Does_Brian_Pass
%   Parameters:
%     M: the transition probability matrix, e.g. brian_matrix()
%     PartyMax:  If Brian parties PartyMax or more times between studying
%       (i.e. state 2) or going to lecture (ie. state 3), he fails.  :(
%       and we return false.  Entering state 6 (i.e. party) or state 8
%       (i.e. party!) counts as a partying.
%     SleepMax:  We use SleepMax as our way to count days.  This is slightly
%       inaccurate because Brian might party all night, or he might fall
%       asleep in lecture.  If Brian makes it to SleepMax with out failing,
%       then he passes.  :)
%     CurrentState:  Brian's current activity, an integer.  For our problem,
%       1 indicates eating, 2 indicates studying, etc.
%     PartyCount: How many times has Brian partied since the last time he
%       studied
%     SleepCount: How many times has Brian slept since the beginning of
%       the term.
%   Result:
%     Does_Brian_Pass: true if Brian passes the simulated term. False if
%       he fails.
pass(M, PartyMax, SleepMax, CurrentState, PartyCount, SleepCount) ->
  NextState = next_state(M, CurrentState),
  if 
	NextState == 2 ->
		pass(M, PartyMax, SleepMax, NextState, 0, SleepCount);
	NextState == 3 ->
		pass(M, PartyMax, SleepMax, NextState, 0, SleepCount);
	NextState == 4 ->
		if
			SleepCount + 1 > SleepMax ->
				true;
			SleepCount + 1 == SleepMax ->
				true;
			true ->
				pass(M, PartyMax, SleepMax, NextState, PartyCount, SleepCount + 1)
		end;
	NextState == 6 ->
		if
			PartyCount + 1 > PartyMax ->
				false;
			PartyCount + 1 == PartyMax ->
				false;
			true ->
				pass(M, PartyMax, SleepMax, NextState, PartyCount + 1, SleepCount)
		end;
	NextState == 8 ->
		if
			PartyCount + 1 > PartyMax ->
				false;
			PartyCount + 1 == PartyMax ->
				false;
			true ->
				pass(M, PartyMax, SleepMax, NextState, PartyCount + 1, SleepCount)
		end;
	true ->
		pass(M, PartyMax, SleepMax, NextState, PartyCount, SleepCount)
	end.

% pass(PartyMax, SleepMax) -> Does_Brian_Pass
%   Wrapper for pass/6 with default parameters for the beginning of the term.
%   Brian starts the term having just slept (i.e. state 4 is 'sleep').  Good
%   thing that he's rested.
pass(PartyMax, SleepMax) -> pass(brian_matrix(), PartyMax, SleepMax, 4, 0, 0).

% pass_count(Ntrials, PartyMax, SleepMax) -> N_Pass
%   Parameters:
%     Ntrials: a non-negative integer.  The number of trials to simulate.
%     PartyMax: If Brian parties PartyMax or more times between studying or
%       going to lecture, he fails that term.
%     SleepMax: If Brian sleeps SleepMax times, we conclude that the term
%       is complete.
%   Result:
%     N_pass: of the Ntrials terms simulated, Brian passed N_pass of them.
pass_count(Ntrials, PartyMax, SleepMax) -> pass_count(Ntrials, PartyMax, SleepMax, 0).
pass_count(0, _, _, NPassed) -> NPassed;
pass_count(Ntrials, PartyMax, SleepMax, NPassed) ->
	Result = pass(PartyMax, SleepMax),
	if	
		Result == true ->
			pass_count(Ntrials - 1, PartyMax, SleepMax, NPassed + 1);
		true ->
			pass_count(Ntrials - 1, PartyMax, SleepMax, NPassed)
	end.

% pass_prob(Ntrials, PartyMax, SleepMax) -> Prob_Pass
%   Parameters: same as for pass/count
%   Result:
%     Prob_Pass: estimated probability of passing, just the fraction of
%       simulated terms for which Brian passes.
pass_prob(Ntrials, PartyMax, SleepMax) ->
  pass_count(Ntrials, PartyMax, SleepMax)/Ntrials.

% pass_prob_par(NProcs, Ntrials, PartyMax, SleepMax) -> Prob_Pass
%   Parallel version of pass_prob.
%   Parameters:
%     NProcs: how many parallel processes to use.
%     Ntrials, PartyMax, SleepMax: same as for pass_count and pass_prob.
%   Notes: I'm so nice that I'm writing the code for the master process.
%     That includes pass_prob_par, pass_spawn, and pass_recv.  You need
%     to write the code for the child processes, pass_worker.
pass_prob_par(NProcs, Ntrials, PartyMax, SleepMax) ->
  Pids = pass_spawn(NProcs, Ntrials, PartyMax, SleepMax),
  pass_recv(Pids, 0, 0).

% pass_spawn(NProcs, Ntrials, PartyMax, SleepMax) -> PidList
%   Spawn NProcs worker tasks and return a list of their pids.
%   Parameters: same as for pass_prob_par.
%   Result: a list of the pids of the worker processes.
%   Notes: we try to divide the Ntrials tasks as evenly as we can among
%     the workers.  That means each worker gets floor(Ntrials/NProcs) or
%     ceil(Ntrials/Nprocs) trials to run.

pass_spawn(0, 0, _PartyMax, _SleepMax) -> [];
pass_spawn(NProcs, Ntrials, PartyMax, SleepMax) ->
  MyPid = self(), % know my pid so I can pass it to the workers.
  MyTrials = Ntrials div NProcs,
  _ = rand:uniform(), % call it once to ensure the default state initialized
  {RandAlgo, _} = rand:export_seed(),
  Seed = list_to_tuple([ rand:uniform(1 bsl 70) || _ <- lists:seq(1,3)]),
  ChildPid = spawn(fun() ->
                     rand:seed_s(RandAlgo, Seed),
		     pass_worker(MyPid, MyTrials, PartyMax, SleepMax)
		   end),
  [ChildPid | pass_spawn(NProcs-1, Ntrials-MyTrials, PartyMax, SleepMax) ].

% pass_recv(PidList, NPassed, Ntrials) -> Prob_Pass
%   Receive results from the worker processes and compute an estimate of
%   the probability of Brian passing the term.
%   Parameters:
%     
pass_recv([], NPassed, NTrials) -> NPassed/NTrials;
pass_recv([Pid | Tl], AccPassed, AccTrials) ->
  receive
    {Pid, Passed, Tried} ->
      pass_recv(Tl, AccPassed+Passed, AccTrials+Tried)
    after 10000 ->
      io:format("Error: time-out waiting for message from ~w.~n", [Pid]),
      io:format("I expected a message of the form {~w, Passed, Tried}~n", [Pid]),
      io:format("My pending messages are:~n"),
      msg_dump(),
      error(time_out)
  end.

% pass_worker(ParentPid, Ntrials, PartyMax, SleepMax)
%   Estimate Brian's probability of passing using
%      pass_count(Ntrials, PartyMax, SleepMax).
%   Send the result to ParentPid with a message of the form
%     {WorkerPid, N_Pass, Ntrials}
%   where WorkerPid is the pid of this worker process, and
%   N_pass is the number of trials for which Brian passed.
pass_worker(ParentPid, Ntrials, PartyMax, SleepMax) ->
  ParentPid ! {self(), Ntrials, pass_count(Ntrials, PartyMax, SleepMax)}.
  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   A few utility functions: msg_dump/0 and your_answer/2                  %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% msg_dump() -> print all pending messages
%   I could use misc:msg_dump from the course library, but I'm trying to
%   minimize that for the first assignment.
msg_dump() ->
  receive
    Msg ->
      io:format("  ~p~n", [Msg]),
      msg_dump()
    after 100 ->
      io:format("  no more messages~n")
  end.

% your_answer(_Ignore, What) -> a place holder for code that you need to write.
%   When you complete your solution, you should have deleted all calls to
%   your_answer(Variables, FunctionAtom) from the templates above and
%   replaced them with your own code.  If you missed any, then this function
%   will get called (if you have a reasonable set of test cases).  It will
%   throw an error to let you know that you still have work to do.
%     Two remarks.  The first argument to your_answer is ignored.  This lets
%   the stubs above compile without warnings.  The calls to your_answer
%   "use" the parameters of the stub functions.  Second, I included this
%   function in the export list so that it will not cause an "unused function"
%   warning when you have completed the homework.
your_answer(_Ignore, What) -> error({incomplete_solution, What}).
