#include "physics/p3/atmosphere_microphysics.hpp"

namespace scream {

void P3Microphysics::run_impl (const int dt)
{
  // Start timer
  auto start = std::chrono::steady_clock::now();

  // Assign values to local arrays used by P3, these are now stored in p3_loc.
  Kokkos::parallel_for(
    "p3_main_local_vals",
    Kokkos::RangePolicy<>(0,m_num_cols),
    p3_preproc
  ); // Kokkos::parallel_for(p3_main_local_vals)
  Kokkos::fence();

  // Update the variables in the p3 input structures with local values.

  infrastructure.dt = dt;
  infrastructure.it++;

  // WorkspaceManager for internal local variables
  const Int nk_pack = ekat::npack<Spack>(m_num_levs);
  const auto policy = ekat::ExeSpaceUtils<KT::ExeSpace>::get_default_team_policy(m_num_cols, nk_pack);
  ekat::WorkspaceManager<Spack, KT::Device> workspace_mgr(m_buffer.wsm_data, nk_pack, 52, policy);

  // Run p3 main
  P3F::p3_main(prog_state, diag_inputs, diag_outputs, infrastructure,
               history_only, workspace_mgr, m_num_cols, m_num_levs);

  // Conduct the post-processing of the p3_main output.
  Kokkos::parallel_for(
    "p3_main_local_vals",
    Kokkos::RangePolicy<>(0,m_num_cols),
    p3_postproc
  ); // Kokkos::parallel_for(p3_main_local_vals)
  Kokkos::fence();

  // Stop timer
  auto finish = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
  const double report_time = 1e-6*duration.count();

  // Compute "Thousands of columns per second"
  const int ncols = m_num_cols;
  const double thousand_cols_per_sec = (ncols/1000.0)/report_time;

  // Print timing
  std::cout << this->name() << ", ncols=" << ncols << ":  Time = " << report_time
            << "        " << "ncol/1000/sec = " << thousand_cols_per_sec << std::endl;
}

} // namespace scream
