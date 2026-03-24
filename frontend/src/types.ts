export type JsonRecord = Record<string, any>;

export interface OverviewResponse {
  counts: JsonRecord;
  training_runs: JsonRecord[];
  testing_runs: JsonRecord[];
  model_versions: JsonRecord[];
  monitoring: JsonRecord;
}
