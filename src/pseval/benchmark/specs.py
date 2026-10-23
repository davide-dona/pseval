from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """How one benchmark is built from its original log.

    The column fields name the original log's case, activity, resource, and timestamp columns;
    `text_columns` names further columns read as text rather than parsed as numbers. The rest
    fixes the protocol: the out-of-time test fraction and the validation fraction drawn at random
    from the train cases with `split_seed`, the percentiles of the outlier and late-start filters,
    and the Declare mining thresholds.
    """

    name: str
    case_column: str
    activity_column: str
    resource_column: str
    timestamp_column: str
    separator: str
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

    def as_dict(self) -> dict[str, object]:
        """The spec as recorded in a bundle manifest."""
        spec = asdict(self)
        spec['text_columns'] = list(self.text_columns)
        return spec


SEPSIS = DatasetSpec(
    name='sepsis',
    case_column='Case ID',
    activity_column='Activity',
    resource_column='Resource',
    timestamp_column='time:timestamp',
    separator=';',
)
BPIC12 = DatasetSpec(
    name='bpic12',
    case_column='CaseID',
    activity_column='Activity',
    resource_column='org:resource',
    timestamp_column='CompleteTimestamp',
    separator=',',
)
BPIC17 = DatasetSpec(
    name='bpic17',
    case_column='case:concept:name',
    activity_column='concept:name',
    resource_column='org:resource',
    timestamp_column='time:timestamp',
    separator=';',
)
BPIC19 = DatasetSpec(
    name='bpic19',
    case_column='case:concept:name',
    activity_column='concept:name',
    resource_column='org:resource',
    timestamp_column='time:timestamp',
    separator=';',
    text_columns=('case:GR-Based Inv. Verif.', 'case:Goods Receipt'),
)

SPECS = {spec.name: spec for spec in (SEPSIS, BPIC12, BPIC17, BPIC19)}
