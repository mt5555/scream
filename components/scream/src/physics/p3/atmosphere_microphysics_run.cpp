#include "physics/p3/atmosphere_microphysics.hpp"

namespace scream {

void P3Microphysics::run_impl (const int dt)
{
  timer_all.start_timer();

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

  timer_wsm.start_timer();
  // WorkspaceManager for internal local variables
  const Int nk_pack = ekat::npack<Spack>(m_num_levs);
  const auto policy = ekat::ExeSpaceUtils<KT::ExeSpace>::get_default_team_policy(m_num_cols, nk_pack);
  ekat::WorkspaceManager<Spack, KT::Device> workspace_mgr(m_buffer.wsm_data, nk_pack, 52, policy);
  timer_wsm.stop_timer();
  wsm_times.push_back(timer_wsm.report_time("wsm",get_comm()));

  timer_main.start_timer();
  // Run p3 main
  P3F::p3_main(prog_state, diag_inputs, diag_outputs, infrastructure,
               history_only, workspace_mgr, m_num_cols, m_num_levs);
  timer_main.stop_timer();
  p3_main_times.push_back(timer_main.report_time("p3_main",get_comm()));

  // Conduct the post-processing of the p3_main output.
  Kokkos::parallel_for(
    "p3_main_local_vals",
    Kokkos::RangePolicy<>(0,m_num_cols),
    p3_postproc
  ); // Kokkos::parallel_for(p3_main_local_vals)
  Kokkos::fence();

  timer_all.stop_timer();
  run_impl_times.push_back(timer_all.report_time("run_impl",get_comm()));
}

} // namespace scream
