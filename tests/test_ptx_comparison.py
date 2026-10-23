"""Check the statistics parser against representative assembly syntax."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_emmy_ptx import ptx_stats, sass_stats


class AssemblyStatisticsTests(unittest.TestCase):
    def test_inline_half_instructions_and_predicates(self):
        stats = ptx_stats('''
            .reg .f32 %f<4>;
            @!%p1 bra DONE;
            st.global.v4.f32 [%rd1], {%f1,%f2,%f3,%f4};
            {mul.f16 %rs1,%rs2,%rs3;
            }
            ret; // comment
        ''')
        self.assertEqual(stats["instruction_count"], 4)
        self.assertEqual(stats["vector_memory_instructions"], 1)
        self.assertEqual(stats["opcodes"]["mul.f16"], 1)

    def test_uniform_predication_and_padding_are_counted_separately(self):
        stats = sass_stats('''
          /*0010*/ @P0 EXIT ; /* 0x000000000000094d */
          /*0020*/ @!UP0 BRA 0x100 ; /* 0x000000000000094d */
          /*0030*/ LDG.E.128 R4, [R2.64] ;
          /*0040*/ NOP ;
        ''')
        self.assertEqual(stats["sass_instruction_count"], 4)
        self.assertEqual(stats["sass_non_nop_count"], 3)
        self.assertEqual(stats["sass_opcodes"]["LDG.E.128"], 1)


if __name__ == "__main__":
    unittest.main()
