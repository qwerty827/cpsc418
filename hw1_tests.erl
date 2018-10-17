-module(hw1_tests).

-include_lib("eunit/include/eunit.hrl").

% Mark added assertClose for checking results when floating point
%   arithmetic is involved.  You can skip ahead and read the tests;
%   you aren't required to be able to read or define Erlang macros,
%   but you'll probably find it handy to be use them.
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


dot_prod_tr_test() ->
  ?assertEqual(23, hw1:dot_prod_tr([2,3], [4,5])),
  ?assertEqual(0, hw1:dot_prod_tr([], [])),
  ?assertError(_, hw1:dot_prod_tr([], [1])),
  ?assertError(_, hw1:dot_prod_tr([1], [])),
  ?assertClose(11, hw1:dot_prod_tr([1,2], [3,4])).
    
matrix_mult_test() ->
  A = [[1, 2], [3, 4]],
  B = [[5, 6], [7, 8]],
  Expect = [[19, 22], [43, 50]], 
  ?assertEqual(Expect, hw1:matrix_mult(A, B)),
  ?assertEqual([[32,37,16],[23,33,17],[36,53,31]],hw1:matrix_mult([[1,3,5],[4,2,3],[6,5,4]],[[1,3,2],[2,3,3],[5,5,1]])),
  ?assertEqual([[181,183],[218,222]],hw1:matrix_mult([[12,11],[15,13]],[[5,7],[11,9]])).

markov_step_test() ->
  M = hw1:brian_matrix(),
  Prob0 = [0.10, 0.35, 0.20, 0.15, 0.09, 0.05, 0.03, 0.03],
  Expect = [0.258, 0.097, 0.108, 0.099, 0.36, 0.024, 0.03, 0.024],
  ?assertClose(Expect, hw1:markov_step(M, Prob0)).

markov_steps_test() ->
  M = hw1:brian_matrix(),
  Prob0 = [0,0,0,1,0,0,0,0],
  N_steps = 3,
  Expect = [0.232, 0.124, 0.116, 0.112, 0.324, 0.0160, 0.060, 0.016],
  ?assertClose(Expect, hw1:markov_steps(M, Prob0, N_steps)).

markov_fixed_point_test() ->
  M = [[0.5, 0.6, 0.3], [0.2, 0.3, 0.3], [0.3, 0.1, 0.4]],
  Prob0 = [0.3, 0.3, 0.4],
  Expect = [0.46987951805145,0.25301204819703,0.27710843375152006],
  ?assertClose(Expect, hw1:markov_fixed_point(M, Prob0, 1.0e-10)).

% close_epsilon() -- error tolerance for close/2.
close_epsilon() -> 1.0e-8.

close(X, X) -> true;
close([Hd1 | Tl1], [Hd2 | Tl2]) ->
  close(Hd1, Hd2) andalso close(Tl1, Tl2);
close(Tuple1, Tuple2) when is_tuple(Tuple1), is_tuple(Tuple2) ->
  close(tuple_to_list(Tuple1), tuple_to_list(Tuple2));
close(X1, X2) when is_number(X1), is_number(X2), (is_float(X1) or is_float(X2)) ->
   abs(X1 - X2) =< close_epsilon()*lists:max([abs(X1), abs(X2), 1]);
close(_, _) -> false.

