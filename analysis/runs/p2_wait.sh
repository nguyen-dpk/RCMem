#!/bin/bash
# exits when every Phase-2 run and replay output exists
cd /home/quang/nguyendpk/LMEE/analysis
fin() { [ $(cat runs/$1/log_*.log 2>/dev/null | grep -c "All scenes finish") -ge $2 ]; }
ok() {
  for r in p2_E0_nofix_s78 p2_E0_nofix_s79 p2_E0_fix_s77 p2_E0_fix_s78 p2_E0_fix_s79 p2_E1_fix_s77 p2_E1_fix_s78 p2_E1_fix_s79 p2_E1b_fix_s77 p2_E1b_fix_s78 p2_E1b_fix_s79; do fin $r 2 || return 1; done
  fin p2_full_E0_fix_s77 4 || return 1
  for r in instr_choice_sub p2_E0_fix_s77 p2_E0_fix_s78 p2_E0_fix_s79; do [ -f out_p2/qa_permute_$r.json ] || return 1; done
}
until ok; do sleep 300; done
echo "PHASE2 GPU WORK DONE"; tail -n 5 runs/p1_queue.log
