from tests.ut.base import TestBase
from vllm_ascend.platform import select_default_all2all_backend


class TestPlatformAll2AllBackend(TestBase):
    def test_select_default_all2all_backend(self):
        cases = (
            ((False, False), "flashinfer_all2allv"),
            ((True, False), "deepep_low_latency"),
            ((False, True), None),
            ((True, True), None),
        )

        for (enable_dbo, enable_sp), expected in cases:
            with self.subTest(enable_dbo=enable_dbo, enable_sp=enable_sp):
                actual = select_default_all2all_backend(enable_dbo=enable_dbo, enable_sp=enable_sp)
                self.assertEqual(actual, expected)
