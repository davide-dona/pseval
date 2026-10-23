from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """How one benchmark is built from its original log.

    `original_sha256` is the hash of the original log the benchmark is built from. The column
    fields name its case, activity, resource, and timestamp columns; `text_columns` names further
    columns read as text rather than parsed as numbers. The rest fixes the protocol: the
    out-of-time test fraction and the validation fraction drawn at random from the train cases
    with `split_seed`, the percentiles of the outlier and late-start filters, and the Declare
    mining thresholds.
    """

    name: str
    case_column: str
    activity_column: str
    resource_column: str
    timestamp_column: str
    separator: str
    original_sha256: str
    text_columns: tuple[str, ...] = ()
    validation_fraction: float = 0.15
    test_fraction: float = 0.2
    split_seed: int = 42
    # Percentile of case length, in events, above which a case is dropped. Measured on the whole
    # log as read, and the length cap of every draw.
    max_length_percentile: float = 98.5
    # Percentile of case duration above which a case is dropped, on the whole log as read.
    max_duration_percentile: float = 99.9
    # Percentile of case duration, after the outlier filters, that a kept case must start at
    # least that long before the end of the log.
    min_observed_duration_percentile: float = 95.0
    declare_min_support: float = 0.05
    declare_min_confidence: float = 1.0
    declare_max_cardinality: int = 2


SEPSIS = DatasetSpec(
    name='sepsis',
    case_column='Case ID',
    activity_column='Activity',
    resource_column='Resource',
    timestamp_column='time:timestamp',
    separator=';',
    original_sha256='2355e82c79b411b3865472bb08e073ed99a4d97880f6a20aa5d25dc7f8307bcb',
)
BPIC12 = DatasetSpec(
    name='bpic12',
    case_column='CaseID',
    activity_column='Activity',
    resource_column='org:resource',
    timestamp_column='CompleteTimestamp',
    separator=',',
    original_sha256='cc781b7ce9422cc88901c35b834beab318b379e0e91ae48a8d34b3cd0b695a7b',
)
BPIC17 = DatasetSpec(
    name='bpic17',
    case_column='case:concept:name',
    activity_column='concept:name',
    resource_column='org:resource',
    timestamp_column='time:timestamp',
    separator=';',
    original_sha256='ca395caee223c6f17c8f65a2f8ddb0eef8cd7c7297c98991b3d63595682d78ab',
)
BPIC19 = DatasetSpec(
    name='bpic19',
    case_column='case:concept:name',
    activity_column='concept:name',
    resource_column='org:resource',
    timestamp_column='time:timestamp',
    separator=';',
    original_sha256='eae54022cecd7f3cc67aec1fb9a7aa2a2c394b9c1b1113b4d86320d12451c8ae',
    text_columns=('case:GR-Based Inv. Verif.', 'case:Goods Receipt'),
)

SPECS = {spec.name: spec for spec in (SEPSIS, BPIC12, BPIC17, BPIC19)}
