-module(hw2_tests).

-include_lib("eunit/include/eunit.hrl").

% functions for question 1: a calculator process
-export([test_tally/2]).
% ([q1b_test/2, q1c_test/2]).
-export([assertCalc/6]).

% functions for question 2: linear functions
%-export([f/2, g/2, h/1]).

% functions for question 3: Parallel calculator
%-export([calc_reduce/3]).  % you need to write this
%-export([calc_seq/2, random_ops/5, calc_ops/0, calc_reduce_test1/0]).
%-export([calc_reduce_test1/1]).

% functions for question 4: Scanning Calculator
%-export([calc_scan_seq/2]).

% other utilities
%-export([close_epsilon/0, close/2]).

% Mark provided assertClose for checking results when floating point
%   arithmetic is involved.
-define(assertClose(Expect, Expr),
        begin
        ((fun () ->
            case hw2:close(Expect, Expr) of
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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
%   Test functions for Q1: a calculator process.                           %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


  
% q1a_test: test a calculator with operations plus, minus, times, divide,
%   set, and exit.
q1a_test() ->
  Calc = hw2:calculator(q1a),
  ?assertClose(5.0,  test_tally(Calc, [{set, 2.0}, {plus,   3.0}])),
  ?assertClose(6.0,  test_tally(Calc, [{set, 2.0}, {times,  3}])),
  ?assertClose(-1.0, test_tally(Calc, [{set, 2.0}, {minus,  3}])),
  ?assertClose(2/3,  test_tally(Calc, [{set, 2.0}, {divide, 3.0}])),
  Calc ! exit.

test_tally(Calc, Ops) ->
  case hw2:calculate(Calc, Ops) of
    ok -> hw2:fetch(Calc);
    Failure -> Failure
  end.

% q1b_test: This time, I'll use an EUnit "test fixture" -- this has a SetUp
%   and CleanUp part in addition to the tests.  For these tests, SetUp spawns
%   the calculator process, CleanUp makes sure the calculator process has
%   terminated, and the tests check the calculator functions.
% The bonus you get for this is that I've provided a macro, ?assertCalc.
%   See the comments explaining what it does below.
% Note: you don't need to use a test fixture.  I did it because it's
%   the "proper" Erlang way to do tests that have state (e.g. the
%   calculator process).  If your test just runs some test cases,
%   that's fine.

% ?assertCalc(WhichCalc, Expected, Ops)
%   Run the specified calculator with the given Ops and check to see if
%     it produces the Expected result.
%   Parameters:
%     WhichCalc: q1a, q1b, or q1c -- which version of the calculator to test.
%     Expected: the expected result
%       When is_number(Expected), assertCalc tests that the calculator's
%         Tally is hw2:close/2 to Expected.
%       When is_list(Expected), assertCalc tests that the calculator's
%         ProcState "close" to Expected.  This means we look at each
%         {Key, ExpectedValue} pair in Expected.  ProcState must have an entry
%         for that Key, and the ActualValue for that entry must be hw2:close/2
%         to the value in ExpectedValue.
%           assertCalc only checks the entries in the calculator's ProcState
%         that are specified in Expected.  In particular, ProcState may have
%         additional entries that are ignored by the test.
%   Why?  Using an Erlang macro gets the module name, line number, and
%     expression string for the test case.  It passes these to the function
%     that actually creates the test.  By using a macro, error messages for
%     failed tests can include more details about any test that fails.
%   Implementation note:
%     assertCalc uses hw2:calculator/1, hw2:calculate/2, hw2:close/2, and
%     hw2:fetch/2.  If you have modified these, then assertCalc may fail
%     in strange and mysterious ways.
-define(assertCalc(WhichCalc, Expected, Ops),
        assertCalc(WhichCalc, Expected, Ops, ?MODULE, ?LINE, ??Ops)).

% q1_test_basic/1, q1_test_procstate/1, and q1_test_fun/1 are test generators.
%   This means that they return a list of tests, where each element of the list
%   is a function that returns a test to run.  We do this because we need to
%   write tests for a calculator process that doesn't exist yet but will exist
%   when the tests are actually run.
%     You don't need to know the details of test generators.  You can just
%   add more tests by adding more elements to the lists, using ?assertCalc
%   to generate the elements.  If you want to learn more about test
%   generators, they are described in "Learn You Some Erlang",
%     https://learnyousomeerlang.com/eunit#test-generators

% q1_test_basic() same tests as for q1a but wrapped up as a lists of test
%   objects so we can reuse them for q1b and q1c.
q1_test_basic(WhichCalc) -> [
  ?assertCalc(WhichCalc, 5.0,  [{set, 2.0}, {plus,   3.0}]),
  ?assertCalc(WhichCalc, 6.0,  [{set, 2.0}, {times,  3}]),
  ?assertCalc(WhichCalc, -1.0, [{set, 2.0}, {minus,  3}]),
  ?assertCalc(WhichCalc, 2/3,  [{set, 2.0}, {divide, 3.0}])
].

% q1_test_procstate() some more tests for q1b and q1c.
%   Check storing and using values in ProcState.
q1_test_procstate(WhichCalc) -> [
  ?assertCalc(WhichCalc, [], [{set, 2.0}, {plus, 3.0}]),
  ?assertCalc(WhichCalc, [{a, 5.0}], [{set, 2.0}, {plus, 3.0}, {store, a}]),
  ?assertCalc(WhichCalc, 51.0,
               [{set, 2.0}, {plus, 3.0}, {store, a}, {set, 7}, {times, 8},
	        {minus, a}]),
  ?assertCalc(WhichCalc, [{a, 5.0}, {b, 51.0}],
	       [{set, 2.0}, {plus, 3.0}, {store, a}, {set, 7}, {times, 8},
	        {minus, a}, {store, b}]),
  ?assertCalc(WhichCalc, [{a, 17.0}, {b, 51.0}],
	       [{set, 2.0}, {plus, 3.0}, {store, a}, {set, 7}, {times, 8},
                {minus, a}, {store, b}, {divide, 3}, {store, a}]),
  % additional tests
  ?assertCalc(WhichCalc, 7.0,
               [{set, 2}, {store, c}, {set, 7}, {times, 1},
	        {minus, d}]),
  ?assertCalc(WhichCalc, [{a, 10.0}], 
		[{set, 2.0}, {plus, 3.0}, {times, 7}, {minus, 5}, {divide, 3}, {store, a}])
].

% q1_test_fun() one more test case for q1c 
%   Calculate with a user provided function.
q1_test_fun(WhichCalc) -> [ % the quadratic equation example from the problem statement
  ?assertCalc(WhichCalc, [{r1, 2.0}, {r2, -19.0}],
    [ [{set,1.0}, {store,a}, {set,17.0}, {store,b}, {set,-38.0}, {store,c}],
      [{set,4}, {times,a}, {times,c}, {store,four_a_c}],
      [ {set,b}, {times,b}, {minus,four_a_c},
	fun(ProcState, Tally) -> {ProcState, math:sqrt(Tally)} end,
	{store, d} ],
      [{set,2}, {times,a}, {store,two_a}],
      [{set,-1}, {times,b}, {plus,d}, {divide,two_a}, {store, r1}],
      [{set,-1}, {times,b}, {minus,d}, {divide,two_a}, {store, r2}] ]),
% additional tests
?assertCalc(WhichCalc, [{a, 12.0}, {b, 36.0}],
	       [{set, 2.0}, {plus, 3.0}, {store, a}, 
		   fun(ProcState, Tally) -> {lists:keyreplace(a, 1, ProcState, {a, 8}), Tally} end,
		   {set, 7}, {times, 8},
		   fun(ProcState, Tally) -> {ProcState, Tally / 2} end,
                {plus, a}, {store, b}, {divide, 3}, {store, a}])
].


% q1b_test_/0 and q1c_test_/0 are test generators.  As described above, test
%   generators return lists of tests.  For q1b_test_ and q1c_test_, we just
%   concatenate the tests described above.  Note that if a function name ends
%   '_test_', EUnit assumes it's a test generator.
q1b_test_() -> q1_test_basic(q1b) ++ q1_test_procstate(q1b).
q1c_test_() -> q1_test_basic(q1c) ++ q1_test_procstate(q1c) ++ q1_test_fun(q1c).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                           %
%    Test functions for Q2:  Linear functions                               %
%                                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

q2_f_test() ->
  ?assertClose(hw2:f([2,3], 4), 11),
  ?assertError(_, hw2:f([2,bananas], 4)).

q2_g_test() ->
  ?assertClose(hw2:f([2,3], hw2:f([4,5], 6)), hw2:f(hw2:g([2,3], [4,5]), 6)),
  % additional tests
  ?assertClose(hw2:f([3.0, 5.0], hw2:f([4,7], 6)), hw2:f(hw2:g([3,5], [4,7]), 6)),
  ?assertNotEqual(hw2:f([5,6], hw2:f([4,7], 6)), hw2:f(hw2:g([4,7], [5,6]), 6)),
  ?assertClose(hw2:g([2,3], [5,6]), [10,15]).
  
q2_h_test() ->
  ?assertClose(3,  hw2:f(hw2:h({set, 3}), 0)),
  ?assertClose(5,  hw2:f(hw2:h({plus, 2}), 3)),
  ?assertClose(12, hw2:f(hw2:h({times, 4}), 3)),
  % additional tests
  ?assertClose(5, hw2:f(hw2:h({minus, 4}), 9)),
  ?assertClose(7, hw2:f(hw2:h({divide, 3}), 21)).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                           %
%    Test functions for Q3:  A parallel calculator                          %
%                                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% assertReduceCalc -- some more macro magic to make test cases easier to write.
% Parameters:
%   Q: which question -- q4 for testing hw2:calc_reduce, q5 for testing hw2:calc_scan
-define(assertReduceCalc(Q, Expected, Ops, Tally0),
        assertReduceCalc(Q, Expected, Ops, Tally0, ?MODULE, ?LINE,
	                 "Ops=" ++ ??Ops ++ ", Tally0=" ++ ??Tally0)).

q3_test_() -> [
  ?assertReduceCalc(q3, 21.0, [{plus, 3}, {times, 14}, {minus, 10}, {divide, 4}, {minus, 5}, {times, 7}], 0),
  ?assertReduceCalc(q3, 70.0, [{plus, 3}, {times, 14}, {minus, 10}, {divide, 4}, {minus, 5}, {times, 7}], 2),
  ?assertReduceCalc(q3, -47.25,
                    [{plus, 3}, {times, 14}, {set, 3}, {minus, 10}, {divide, 4}, {minus, 5}, {times, 7}],
		    1234.5),
  % additional tests
  ?assertReduceCalc(q3, 16.0,[{divide, 4}, {plus, 5}, {times, 7}, {minus, 11}, {divide, 5}], 32),
  ?assertReduceCalc(q3, 18.0,[{times, 12}, {plus, 5}, {divide, 5}, {minus, 11}, {times, 9}], 5)  
].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                           %
%    Functions for Q4: a scanning calculator                                %
%                                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

q4_test_() -> [
  ?assertReduceCalc(q4, [15, 210, 3, -7, -1.75, -6.75, -47.25],
                    [{plus, 3}, {times, 14}, {set, 3}, {minus, 10}, {divide, 4}, {minus, 5}, {times, 7}],
		    12),
  ?assertReduceCalc(q4, [60, 65, 13, 2, 18],
                    [{times, 12}, {plus, 5}, {divide, 5}, {minus, 11}, {times, 9}],
		    5),
  ?assertReduceCalc(q4, [8, 13, 91, 80, 16],
                    [{divide, 4}, {plus, 5}, {times, 7}, {minus, 11}, {divide, 5}],
		    32)
].


%calc_scan_seq(OpList, Tally0) ->
%  {Result, _} = lists:mapfoldl(
%    fun(Op, Tally) ->
%      NewTally = f(h(Op), Tally),
%      {NewTally, NewTally}
%    end,
%    Tally0,
%    OpList),
%  Result.


% close_keylist(Expected, Actual) -- check the values in a keylist
%   Expected and Actual are both keylists where each entry is a tuple of
%     the form {Key, Value}.
%   close_keylist returns 'true' if each entry in Expected has a corresponding
%     entry in Actual, and the values for these entries are close.
%   Otherwise, close_keylist returns a list of errors, ErrorList:
%     ErrorList includes an entry of the form  '{wrong_entry, {Key, ActualValue}}'
%       if Key is a key for Expected, there is an entry, ActualValue,
%       for Key in Actual, and the entry is not hw2:close/2 to the value
%       from Expected.
%     ErrorList includes an entry of the form '{missing_entry Key}' if Key
%       is a key for Expected, and there is no entry for Key in Actual.
close_keylist(Expected, Actual) ->
  case close_keylist_help(Expected, Actual) of
    [] -> true;
    Failure -> Failure
  end.

close_keylist_help([{Key, Value} | Tl], Actual) ->
  case lists:keyfind(Key, 1, Actual) of
    {Key, V} ->
      case hw2:close(Value, V) of
        true -> [];
	false -> [{wrong_entry, {Key, V}}]
      end;
    false -> [{missing_entry, Key}]
  end ++ close_keylist_help(Tl, Actual);
close_keylist_help([], _) -> [].

% test_calc: send one more commands to a calculator process, and
%   check the Tally or ProcState the end to see if it matches Expected.
%   If the test passes, test_calc returns true.
%   Otherwise, test_calc returns a non-empty list of errors.
test_calc(Calc, Expected, Ops) ->
  case hw2:calculate(Calc, Ops) of
    ok when is_number(Expected) ->  % check the calculator's Tally
      Tally = hw2:fetch(Calc, tally),
      case hw2:close(Expected, Tally) of
	true  -> true;  % the test passed
	false -> [{value, Tally}]
      end;
    ok when is_list(Expected) -> % check the calculator's ProcState
      CalcState = hw2:fetch(Calc, procState),
      case close_keylist(Expected, CalcState) of
        true -> true; % the test passed
	Errors -> [{value, CalcState} | Errors]
      end;
    {dead, Calc} -> {dead_calculator}
  end.


assertCalc(WhichCalc, Expected, Ops, Module, LineNumber, OpsStr) ->
  SetUp = fun() -> hw2:calculator(WhichCalc) end, % start the calculator process
  CleanUp = fun(Calc) ->  % terminate the calculator process
    Calc ! exit,
    exit(Calc, kill) % just making sure
  end,
  Test = fun(Calc) ->
    ?_test(
      case test_calc(Calc, Expected, Ops) of
	true -> ok;
	Failure ->
	  erlang:error({assertCalc_failed,
		        [{module, Module},
			 {line, LineNumber},
			 {expression, OpsStr},
			 {expected, Expected}
			 | Failure ]})
      end)
  end,
  {setup, SetUp, CleanUp, Test}.


assertReduceCalc(Q, Expected, Ops, Tally0, Module, LineNumber, OpsStr) ->
  SetUp = fun() -> wtree:create(4) end,
  CleanUp = fun(W) ->  % terminate the worker-tree processes
    wtree:reap(W)
  end,
  Eval = fun(W) ->
    workers:update(W, ops, misc:cut(Ops, W)),
    case Q of
      q3 -> hw2:calc_reduce(W, ops, Tally0);
      q4 ->
        hw2:calc_scan(W, ops, dst, Tally0),
	lists:append(workers:retrieve(W, dst))
    end
  end,
  Test = fun(W) ->
    ?_test(
      case hw2:close(Expected, Actual=Eval(W)) of
	true  -> ok;
	false ->
	  erlang:error({assertReduceCalc_failed,
		        [{module, Module},
			 {line, LineNumber},
			 {expression, OpsStr},
			 {expected, Expected},
			 {value, Actual}]})
      end)
  end,
  {setup, SetUp, CleanUp, Test}.
