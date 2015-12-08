<?
function page_rank_vd_Constant_State(array $t_args)
{
    $className          = $t_args['className'];
?>
using namespace std;

class <?=$className?>ConstantState {
 private:
  // The current iteration.
  int iteration;

  // The number of distinct nodes in the graph.
  long nVertices;

 public:
  friend class <?=$className?>;

  <?=$className?>ConstantState()
      : iteration(0),
        nVertices(0) {
  }
};

<?
    return [
        'kind' => 'RESOURCE',
        'name' => $className . 'ConstantState',
    ];
}

function page_rank_vd($t_args, $inputs, $outputs)
{
    $className = generate_name('PageRank');

    $inputs_ = array_combine(['s', 't'], $inputs);
    $vertex = $inputs_['s'];

    $outputs_ = ['node' => $vertex, 'rank' => lookupType('double')];
    $outputs = array_combine(array_keys($outputs), $outputs_);

    $sys_headers  = ['vector', 'algorithm'];
    $user_headers = [];
    $lib_headers  = [];
    $libraries    = [];
    $properties   = [];
    $extra        = [];
    $result_type  = ['fragment'];
?>

using namespace std;

class <?=$className?>;

<?  $constantState = lookupResource(
        'pagerankVD::page_rank_vd_Constant_State',
        ['className' => $className]
    ); ?>

class <?=$className?> {
 public:
  using ConstantState = <?=$constantState?>;
  using Iterator = std::pair<int, int>;
  static const constexpr float kDamping = 0.85;
  static const constexpr int nIterations = 5;
  static const constexpr int kBlock = 32;
  static const constexpr int kMaxFragments = 64;

 private:

  static std::vector<double> weight;
  static std::vector<double> pagerank;
  static std::vector<double> oldPagerank;

  const ConstantState& constant_state;
  long nVertices;
  int iteration;
  int num_fragments;

 public:
  <?=$className?>(const <?=$constantState?>& state)
      : constant_state(state),
        nVertices(state.nVertices),
        iteration(state.iteration) {
  }

  void AddItem(<?=const_typed_ref_args($inputs_)?>) {
    if (iteration == 0) {
      nVertices = max((long) max(s, t), nVertices);
      return;
    } else if (iteration == 1) {
      weight[s]++;
    } else {
      oldPagerank[t] += weight[s] * pagerank[s];
    }
  }

  void AddState(<?=$className?> &other) {
    if (iteration == 0)
      nVertices = max(nVertices, other.nVertices);
  }

  // Most computation that happens at the end of each iteration is parallelized
  // by performed it inside Finalize.
  bool ShouldIterate(ConstantState& state) {
    state.iteration = ++iteration;
    cout << "Current Iteration: " << iteration;
      if (iteration == 1) {
      state.nVertices = ++nVertices;
      cout << "num_nodes: " << num_nodes << endl;
      oldPagerank.reserve(nVertices);
      pagerank.reserve(nVertices);
      weight.reserve(nVertices);
      std::fill (pagerank.begin(),pagerank.end(),1);
      return true;
    } else {
      return iteration < nIterations + 1;
    }
  }

  int GetNumFragments() {
    cout << "Returning " << num_fragments << " fragments" << endl;
    long size = (nVertices - 1) / kBlock + 1;  // nVertices / kBlock rounded up.
    num_fragments = (iteration == 0) ? 0 : min(size, (long) kMaxFragments);
    return num_fragments;
  }

  Iterator* Finalize(long fragment) {
    int count = nVertices;
    int first = fragment * (count / kBlock) / num_fragments * kBlock;
    int final = (fragment == num_fragments - 1) ? count - 1 : (fragment + 1) * (count / kBlock) / num_fragments * kBlock - 1;
    if  (iteration == 2) {
      for(int i = first; i <= final; i++){
        pagerank[i] = 1;
        weight[i] = 1/weight[i];
        oldPagerank[i] = 0;
      }
    } else {
      for(int i = first; i <= final; i++){
        pagerank[i] = (1 - kDamping) + kDamping * oldPagerank[i];
        oldPagerank[i] = 0;
      }
    }
    return new Iterator(first, final);
  }

  bool GetNextResult(Iterator* it, <?=typed_ref_args($outputs_)?>) {
    if (iteration < nIterations + 1)
      return false;
    if (it->first > it->second)
      return false;
    node = it->first;
    rank = pagerank[it->first];
    it->first++;
    return true;
  }
};

// Initialize the static member types.
vector<double> <?=$className?>::weight;
vector<double> <?=$className?>::pagerank;
vector<double> <?=$className?>::oldPagerank;

typedef <?=$className?>::Iterator <?=$className?>_Iterator;

<?
    return [
        'kind'            => 'GLA',
        'name'            => $className,
        'system_headers'  => $sys_headers,
        'user_headers'    => $user_headers,
        'lib_headers'     => $lib_headers,
        'libraries'       => $libraries,
        'properties'      => $properties,
        'extra'           => $extra,
        'iterable'        => true,
        'intermediates'   => true,
        'input'           => $inputs,
        'output'          => $outputs,
        'result_type'     => $result_type,
        'generated_state' => $constantState,
    ];
}
?>
