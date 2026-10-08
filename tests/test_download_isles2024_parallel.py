from scripts.download_isles2024_parallel import plan_ranges


def test_plan_ranges_covers_exact_interval_without_overlap():
    ranges = plan_ranges(start=100, total_size=350, part_size=100)

    assert [(item.start, item.end) for item in ranges] == [
        (100, 199),
        (200, 299),
        (300, 349),
    ]
    assert sum(item.size for item in ranges) == 250


def test_plan_ranges_empty_when_prefix_is_complete():
    assert plan_ranges(start=1024, total_size=1024, part_size=256) == []
