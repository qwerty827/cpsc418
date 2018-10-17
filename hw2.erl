-module(hw2).

% functions for question 1: a calculator process
-export([calculator/1, q1a/1, q1b/2, q1c/2]).
-export([fetch/1, fetch/2, calculate/2, solve_quad/3, solve_quad/4]).

% functions for question 2: linear functions
-export([f/2, g/2, h/1]).

% functions for question 3: Parallel calculator
-export([calc_reduce/3]).  % you need to write this
-export([calc_seq/2, random_ops/5, calc_ops/0, calc_reduce_test1/0]).
-export([calc_reduce_test1/1]).

% functions for question 4: Scanning Calculator
-export([calc_scan_seq/2]).
-export([calc_ser/2, calc_helper/2, calc_merge/3, calc_scan/4]).

% other utilities
-export([close_epsilon/0, close/2]).
-export([your_answer/2]).


% Mark provided assertClose for checking results when floating point
%   arithmetic is involved.
-define(assertClose(Expect, Expr),
        begin
        ((fun () ->
            case close(Expect, Expr) of
                true  -> ok;
                false -> erlang:error({assertClose,
                                     [{module, ?MODULE},
                                      {line, ?LINE},
                                      {expression, (??Expr)},
                                      {expected, Expect},
                                      {value, Expr}]})
            end
          end)())
        end).

% close_epsilon() -- error tolerance for close/2.
close_epsilon() -> 1.0e-10.

close(X, X) -> true;
close([Hd1 | Tl1], [Hd2 | Tl2]) ->
  close(Hd1, Hd2) andalso close(Tl1, Tl2);
close(Tuple1, Tuple2) when is_tuple(Tuple1), is_tuple(Tuple2) ->
  close(tuple_to_list(Tuple1), tuple_to_list(Tuple2));
close(X1, X2) when is_number(X1), is_number(X2), (is_float(X1) or is_float(X2)) ->
   abs(X1 - X2) =< close_epsilon()*lists:max([abs(X1), abs(X2), 1]);
close(_, _) -> false.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q1: a calculator process.                                %
%     You need to complete functions q1a, q1b, q1b_op, and q1c.            %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

calculator(Q) ->
  case Q of
    q1a -> spawn(fun() -> q1a(0) end);
    q1b -> spawn(fun() -> q1b([], 0) end);
    q1c -> spawn(fun() -> q1c([], 0) end)
  end.

q1a(Tally) when is_number(Tally) ->
  receive
    {plus, X}     when is_number(X) -> q1a(Tally+X);
    {minus, X}    when is_number(X) -> q1a(Tally-X);
    {times, X}    when is_number(X) -> q1a(Tally*X);
    {divide, X}   when is_number(X) -> q1a(Tally/X);
    {set, X}      when is_number(X) -> q1a(X);
    {tally, Pid} when is_pid(Pid)  ->
      Pid ! {tally, self(), Tally},
      q1a(Tally);
    exit -> Tally
  end.

q1b(ProcState, Tally) when is_number(Tally) ->
  receive
    {tally, Pid} when is_pid(Pid)  -> 
	  Pid ! {tally, self(), Tally},
	  q1b(ProcState, Tally);
    {procState, Pid} when is_pid(Pid)  ->
      % send our ProcState to Pid -- this let's you peak at ProcState to
      %   better understand how it works.
      Pid ! {procState, self(), ProcState},
      q1b(ProcState, Tally);
    {store, Key} when is_atom(Key) ->
      % Add an entry to ProcState whose key is Key and whose value is the current Tally.
      %   See the documentation for lists:keyfind/3.
      %   ProcState should be a tuple list where each tuple has two elements.
      %   The first element is the key, the second element is the value.
	  CurrTuple = lists:keyfind(Key, 1, ProcState),
	  if 
		is_tuple(CurrTuple) ->
		  NextProcState = lists:keyreplace(Key, 1, ProcState, {Key, Tally}),
		  q1b(NextProcState, Tally);
		true ->
		  NextProcState = lists:append(ProcState, [{Key, Tally}]),
		  q1b(NextProcState, Tally)
	  end;
    {Op, X} ->
      {NewProcState, NewTally} = q1b_op(ProcState, Tally, Op, X),
      q1b(NewProcState, NewTally);
    exit -> Tally
  end.

q1b_op(ProcState, Tally, Op, X) ->
  % XX = value for X.
  % If X is a number, that's the value.
  % If X is an atom, it's a key.
  %   Look up the corresponding value in ProcState.
  %   If an entry is found, XX gets the corresponding value.
  %   Otherwise, print the error message
  %       io:format("~w: no value defined for ~w~n", [self(), X])
  %     and set XX to the atom 'fail'.
  XX =
	if 
		is_number(X) -> X;
		is_atom(X) -> 
			T = lists:keyfind(X, 1, ProcState),
			if 
			  is_tuple(T) -> element(2, T);
			  true -> io:format("~w: no value defined for ~w~n", [self(), X]),
			  fail
			end;
		true -> io:format("~w: no value defined for ~w~n", [self(), X]),
				fail
	end,
  case {Op, XX} of
    {_, fail}    -> {ProcState, Tally}; % keep going, but don't change anything
    {plus, XX}   -> {ProcState, Tally+XX};
    {minus, XX}  -> {ProcState, Tally-XX};
    {times, XX}  -> {ProcState, Tally*XX};
    {divide, XX} -> {ProcState, Tally/XX};
    {set, XX}    -> {ProcState, XX}
  end.

q1c(ProcState, Tally) when is_number(Tally) ->
	receive
    {tally, Pid} when is_pid(Pid)  -> 
	  Pid ! {tally, self(), Tally},
	  q1c(ProcState, Tally);
    {procState, Pid} when is_pid(Pid)  ->
      % send our ProcState to Pid -- this let's you peak at ProcState to
      %   better understand how it works.
      Pid ! {procState, self(), ProcState},
      q1c(ProcState, Tally);
    {store, Key} when is_atom(Key) ->
      % Add an entry to ProcState whose key is Key and whose value is the current Tally.
      %   See the documentation for lists:keyfind/3.
      %   ProcState should be a tuple list where each tuple has two elements.
      %   The first element is the key, the second element is the value.
	  CurrTuple = lists:keyfind(Key, 1, ProcState),
	  if 
		is_tuple(CurrTuple) ->
		  NextProcState = lists:keyreplace(Key, 1, ProcState, {Key, Tally}),
		  q1b(NextProcState, Tally);
		true ->
		  NextProcState = lists:append(ProcState, [{Key, Tally}]),
		  q1c(NextProcState, Tally)
	  end;
    {Op, X} ->
      {NewProcState, NewTally} = q1b_op(ProcState, Tally, Op, X),
      q1c(NewProcState, NewTally);
	F when is_function(F, 2) ->
	  {NewProcState, NewTally} = F(ProcState, Tally),
	  q1c(NewProcState, NewTally);
    exit -> Tally
	end.
	  

% some helpful functions for debugging and testing the calculator questions

% fetch(Pid, What, TimeOut) -> Value
%   Retrieve a value from the calculator process Pid.
%     Pid: the Pid of the calculator process.
%     What:
%       If What is 'tally', we send the current Tally, a number.
%       If What is 'procState', we send the current ProcState, a key-list.
%     TimeOut -> give up and print an error message if no response.
%       If TimeOut is an integer, it is the maximum time to wait in milliseconds.
%       If TimeOut is a float, it is the maximum time to wait in seconds.
%
%   Note: wrapping up the messaging with a function is an example of
%     "keeping messages secret" as described in LYSE.
fetch(Pid, What, TimeOut) when is_integer(TimeOut) ->
  Pid ! {What, self()},
  receive {What, Pid, V} -> V
  after TimeOut ->
    misc:msg_dump([io_lib:format("{~w, ~w, V}", [What, Pid])])
  end;
fetch(Pid, What, TimeOut) when is_float(TimeOut) ->
  fetch(Pid, What, round(1000*TimeOut)).

% abbreviated calls to fetch with default for What and/or TimeOut.
fetch(Pid, What) when is_atom(What) -> fetch(Pid, What, 100);
fetch(Pid, TimeOut) when is_number(TimeOut) -> fetch(Pid, tally, TimeOut).
fetch(Pid) -> fetch(Pid, 100).

% sending one message for each operation gets tedious.
%   calculate traverses a deep-list of operations and sends them to
%   Calculator.
%   At the end, we check to make sure the Calculator process hasn't died.
%     If Calculator is still alive, we return 'ok'.
%     Otherwise, we return {dead, Calculator}.
calculate(Calculator, []) when is_pid(Calculator) -> calculator_ok(Calculator);
calculate(Calculator, [Op | Tl]) when is_pid(Calculator) ->
  calculate(Calculator, Op),
  calculate(Calculator, Tl);
calculate(Calculator, Op) when is_pid(Calculator) ->
  Calculator ! Op,
  calculator_ok(Calculator).

calculator_ok(Calculator) when is_pid(Calculator) ->
  case process_info(Calculator, status) of
    {status,_} -> ok;
    undefined -> {dead, Calculator}
  end.

% Solve the quadratic equation A*X*X + B*X + C = 0 using the calculator
%   from q1c.
solve_quad(Calculator, A, B, C) ->
  ok = calculate(Calculator,
         [ [{set,A}, {store,a}, {set,B}, {store,b}, {set,C}, {store,c}],
           [{set,4}, {times,a}, {times,c}, {store,four_a_c}],
           [ {set,b}, {times,b}, {minus,four_a_c},
	     fun(ProcState, Tally) -> {ProcState, math:sqrt(Tally)} end,
	     {store, d} ],
	   [{set,2}, {times,a}, {store,two_a}],
           [{set,-1}, {times,b}, {plus,d}, {divide,two_a}, {store, r1}],
           [{set,-1}, {times,b}, {minus,d}, {divide,two_a}, {store, r2}] ]),
  Get = fun(What) -> Calculator ! {set, What}, fetch(Calculator) end,
  {Get(r1), Get(r2)}.
  
% solve_quad(A, B, C):
%   Same as solve_quad/4 but we spawn the calculator and send it an exit
%   message when we're done.  I've included try/catch code so we make a
%   reasonable effort to clean up if something fails.
solve_quad(A, B, C) ->
  Calculator = calculator(q1c),
  try
    Ans = solve_quad(Calculator, A, B, C),
    Calculator ! exit,
    Ans
  catch _:Reason ->
    Calculator ! exit, % clean up
    error(Reason) % rethrow
  end.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q2: linear functions.                                    %
%     You need to complete functions g and h                               %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f([A, B], X) -> A*X + B.

g([A2, B2], [A1, B1]) -> [A1*A2, A2*B1+B2].

% h: translate calculator messages to linear operators
%   You may rewrite h({Op, X}) using one pattern for each operator --
%   that's what my solution does.  h needs to accept {plus, X},
%   {minus, X}, {times, X}, {divide, X}, {set, X}.
h({plus, X}) -> [1, X];
h({minus, X}) -> [1, -X];
h({times, X}) -> [X, 0];
h({divide, X}) -> [1/X, 0];
h({set, X}) -> [0, X];
h({_, _}) -> error("invalid argument").


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q3: a parallel calculator.                               %
%     You need to write the implementation of calc_reduce and add test     %
%     cases to hw2_tests.erl.  You also need to make some timing           %
%     measurements.  You can write one or more functions here to make      %
%     those measurements, and/or you can give commands to the Erlang.      %
%     Either way, you should use the time_it:t function to measure time    %
%     and your write-up in hw2.pdf should clearly state how you made your  %
%     measurements.                                                        %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

calc_leaf([]) -> undef;
calc_leaf({Op, X}) -> h({Op, X});
calc_leaf([Hd | Tl]) -> calc_leaf(Tl, calc_leaf(Hd)).
calc_leaf([], T) -> T;
calc_leaf([Hd | Tl], T) -> calc_leaf(Tl, g(calc_leaf(Hd), T)).

calc_combine(undef, Right) -> Right;
calc_combine(Left, undef) -> Left;
calc_combine(Left, Right) -> g(Right, Left).

calc_reduce(W, Key, Tally0) -> 
	Leaf = fun(ProcState) ->
				Data = wtree:get(ProcState, Key),
				calc_leaf(Data)
			end,
	Combine = fun(Left, Right) ->
				calc_combine(Left, Right)
			end,
	Root = fun(Acc) ->
				f(Acc, Tally0)
			end,
	wtree:reduce(W, Leaf, Combine, Root).

% calc_reduce_test1, calc_seq, random_ops, and calc_ops are provided to
%   allow you to get started faster.
calc_reduce_test1(Tally0) ->
  Ops = calc_ops(),
  W = wtree:create(4),
  workers:update(W, ops, misc:cut(Ops, W)),
  Ans_par = calc_reduce(W, ops, Tally0),
  Ans_seq = calc_seq(Ops, Tally0),
  case close(Ans_par, Ans_seq) of
    true -> Ans_par;
    false -> {{got, Ans_par}, {expected, Ans_seq}}
  end.

calc_reduce_test1() -> calc_reduce_test1(0).

calc_ops() ->
  [{plus, 3}, {times, 14}, {minus, 10}, {divide, 4}, {minus, 5}, {times, 7}].


% calc_seq(Ops, X) -> Result
%   Apply the operations of Ops in left to right order, starting with
%     the value of X.
%   For example:
%     calc_seq([{plus, 3}, {times, 14}], 0) ->
%       calc_seq([{times, 14}], 0+3) ->
%         calc_seq([{times, 14}], 3) ->
%         calc_seq({times, 14}, 3) -> 14*3 -> 42
%   Ops can be:
%     A tuple of the form {Op, Y} where Op is one the following five atoms:
%       plus, minus, times, divide, set; or
%     A list of Ops.  In other words, a list of tuples, or a nested list
%       of tuples.
%   X must be a number.
calc_seq([], X) when is_number(X) -> X;
calc_seq([Hd | Tl], X) -> calc_seq(Tl, calc_seq(Hd, X));
calc_seq(Op, X) -> f(h(Op), X).

random_ops(W, N, OpsAllowed, _M, Key) ->
  WorkerArgs =
    [Hi-Lo || {Lo,Hi} <- misc:intervals(0, N, workers:nworkers(W))],
  workers:broadcast(W,
    fun(ProcState0, NN) ->
      {I, ProcState1} = workers:random(NN, length(OpsAllowed), ProcState0),
      {V, ProcState2} = workers:random(NN, 1.0, ProcState1),
      workers:put(ProcState2, Key,
        lists:zip([ lists:nth(II, OpsAllowed) || II <- I ],
                  [ 0.45*((0.8/0.45)+VV) || VV <- V]))
    end, WorkerArgs).





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Functions for Q4: a scanning calculator                                %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

calc_scan_seq(OpList, Tally0) ->
  {Result, _} = lists:mapfoldl(
    fun(Op, Tally) ->
      NewTally = f(h(Op), Tally),
      {NewTally, NewTally}
    end,
    Tally0,
    OpList),
  Result.
  
 
calc_ser(SrcLst) -> calc_helper(SrcLst, maps:new()).
calc_helper([], CurrMap) ->
    CurrMap;
calc_helper([H | T], CurrMap) -> 
	NewMap = maps:update_with(op,
						fun(OldOp) -> g(h(H), OldOp) end,
						h(H),
						NewMap),
    calc_helper(T, NewMap).

calc_merge(Left, Right) ->
	LeftOp = maps:get(op, Left),
	RightOp = maps:get(op, Right),
	NewOp = g(RightOp, LeftOp),
	#{op => NewOp}.

calc_scan(W, SrcKey, DstKey, Tally0) ->
    wtree:scan(W,
				fun(ProcState) -> 
					SrcList = workers:get(ProcState, SrcKey),
					calc_ser(SrcList)
				end,
				fun(ProcState, AccIn) ->
					SrcList = workers:get(ProcState, SrcKey),
					ResMap = calc_helper(SrcList, AccIn),
					ResOp = maps:get(op, ResMap),
					Result = f(ResOp, Tally0),
					workers:put(ProcState, DstKey, Result) end,
				fun(Left, Right) ->
					calc_merge(Left, Right) end,
				maps:new().
					).

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
