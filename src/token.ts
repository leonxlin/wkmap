export interface Token {
  name: string;
  vector: number[];
  vectorNormed?: number[];
  norm?: number;

  // Index in source data (which may be sorted by frequency/importance).
  index: number;

  // Properties related to the html element representing this record.
  node?: Element;
  plotPos: number[];
}
