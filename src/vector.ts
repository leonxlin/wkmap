import * as d3 from "d3";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";

import { Token } from "./token";

function writeNormedVectorsToTokens(
  data: Token[],
  vectorNorms: tf.Tensor2D,
  vectorsNormed: tf.Tensor2D
) {
  const vnds = vectorsNormed.arraySync() as number[][];
  const vns = tf.util.flatten(vectorNorms.arraySync()) as number[];
  for (let i = 0; i < vnds.length; ++i) {
    data[i].vectorNormed = vnds[i];
    data[i].norm = vns[i];
  }
}

// TODO: separate word lookup from projection computation.
function computeWordPairProjection(
  wordA: string,
  wordB: string,
  data: Token[],
  vectors: tf.Tensor2D
): tf.Tensor2D {
  let vecA!: tf.Tensor, vecB!: tf.Tensor;
  data.forEach((d) => {
    if (d.name == wordA && d.vectorNormed) {
      vecA = tf.tensor(d.vectorNormed);
    } else if (d.name == wordB && d.vectorNormed) {
      vecB = tf.tensor(d.vectorNormed);
    }
  });

  const N = data.length;

  if (!vecA || !vecB) {
    throw new Error(`Vector not found for ${wordA} or ${wordB} or both`);
    console.log(`Vector not found for ${wordA} or ${wordB} or both`);
  }

  const vecAB = tf.sub(vecB, vecA);
  const vecAOs = tf.sub(vectors, tf.reshape(vecA, [1, 300]));
  const sims = tf.div(
    tf.matMul(vecAOs, tf.reshape(vecAB, [300, 1])),
    tf.dot(vecAB, vecAB)
  ) as tf.Tensor2D;
  const distancesToABLine = tf.norm(
    tf.sub(
      vecAOs,
      tf.matMul(tf.reshape(sims, [N, 1]), tf.reshape(vecAB, [1, 300]))
    ),
    /*ord=*/ 2,
    /*axis=*/ 1,
    /*keepDims=*/ true
  ) as tf.Tensor2D;

  return tf.concat2d([sims, distancesToABLine], /*axis=*/ 1);
}

export class VectorManager {
  constructor(
    public tokens: Token[],

    // Tensors storing all token vectors for easy matrix operations.
    // Dimension 0: tokens.
    // Dimension 1: Vector embedding components.
    // TODO: it's probably not necessary to store the vectors in both `tokens` and `vectors`.
    private vectors: tf.Tensor2D,
    private vectorNorms: tf.Tensor2D, // TODO: figure out if this should be flattened.
    private vectorsNormed: tf.Tensor2D
  ) {}

  setTokenPlotPositions(coords: number[][]): Token[] {
    if (!this.tokens) return [];
    this.tokens.forEach((d, index) => {
      d.plotPos = coords[index];
    });
    return this.tokens;
  }

  setTokenPlotPositionsFromFn(
    getX: (d: Token) => number,
    getY: (d: Token) => number
  ): Token[] {
    if (!this.tokens) return [];
    this.tokens.forEach((d) => {
      d.plotPos = [getX(d), getY(d)];
    });
    return this.tokens;
  }

  projectToComponents(component1: number, component2: number): Token[] {
    if (!this.tokens) return [];
    this.tokens.forEach((d) => {
      if (d.vectorNormed) {
        d.plotPos = [d.vectorNormed[component1], d.vectorNormed[component2]];
      } else {
        throw new Error(`vectorNormed not found for ${d.name}`);
      }
    });
    return this.tokens;
  }

  // tokenNamesA and tokenNamesB must have the same length.
  projectToTokenPairsAxis(
    tokenNamesA: string[],
    tokenNamesB: string[]
  ): Token[] {
    if (!this.tokens) return [];

    const projs: tf.Tensor2D[] = [];
    for (let i = 0; i < tokenNamesA.length; ++i) {
      projs.push(
        computeWordPairProjection(
          tokenNamesA[i],
          tokenNamesB[i],
          this.tokens,
          this.vectorsNormed
        )
      );
    }
    const coords = tf.div(tf.addN(projs), tokenNamesA.length) as tf.Tensor2D;
    const coordsArr = coords.arraySync() as number[][];

    this.setTokenPlotPositions(coordsArr);
    return this.tokens;
  }

  // tokenNamesA and tokenNamesB need not have the same length.
  projectToTokenPairsAxis2(
    tokenNamesA: string[],
    tokenNamesB: string[]
  ): Token[] {
    if (!this.tokens) return [];

    const vecsA = [] as tf.Tensor2D[],
      vecsB = [] as tf.Tensor2D[];
    this.tokens.forEach((d) => {
      if (tokenNamesA.includes(d.name) && d.vectorNormed) {
        vecsA.push(tf.tensor2d(d.vectorNormed, [300, 1]));
      } else if (tokenNamesB.includes(d.name) && d.vectorNormed) {
        vecsB.push(tf.tensor2d(d.vectorNormed, [300, 1]));
      }
    });

    const matA = tf.concat2d(vecsA, /*axis=*/ 1);
    const matB = tf.concat2d(vecsB, /*axis=*/ 1);
    const simsA = tf.matMul(this.vectorsNormed, matA);
    const simsB = tf.matMul(this.vectorsNormed, matB);
    const aggDistA = tf.exp(
      tf.mean(
        tf.log(tf.maximum(tf.mul(tf.sub(1, simsA), 0.5), 0)),
        /*axis=*/ 1,
        /*keepDims=*/ true
      )
    );
    const aggDistB = tf.exp(
      tf.mean(
        tf.log(tf.maximum(tf.mul(tf.sub(1, simsB), 0.5), 0)),
        /*axis=*/ 1,
        /*keepDims=*/ true
      )
    );

    const totDist = tf.add(aggDistA, aggDistB);
    const abScale = tf.softmax(tf.concat([aggDistA, aggDistB], /*axis=*/ 1));
    const coords = tf.concat([abScale, totDist], /*axis=*/ 1);
    const coordsArr = coords.arraySync() as number[][];

    this.setTokenPlotPositionsFromFn(
      (d) => {
        return coordsArr[d.index][0];
      },
      (d) => {
        return coordsArr[d.index][2];
      }
    );
    return this.tokens;
  }

  // Returns 10 closest neighbors according to cosine similarity of normed vectors.
  getNeighbors(token: Token): Token[] {
    if (!this.tokens) return [];

    const similarities = tf.util.flatten(
      tf
        .matMul(
          this.vectorsNormed,
          tf.tensor2d(token.vectorNormed || token.vector, [300, 1])
        )
        .arraySync()
    ) as number[];
    const sim10 = [...similarities].sort(d3.descending)[10];

    return this.tokens.filter((d) => similarities[d.index] >= sim10);
  }
}

export async function loadVectors(
  path: string,
  skipHeader = false
): Promise<VectorManager> {
  const raw = await d3.text(path);
  const dsv = d3.dsvFormat(" ");
  let rows = dsv.parseRows(raw);
  if (skipHeader) {
    rows = rows.slice(1);
  }

  const tokens = rows.map((row, index) => {
    return {
      name: row[0],
      vector: row.slice(1).map((a) => parseFloat(a)),
      index: index,
      plotPos: [0, 0],
    };
  });

  const vectors = tf.tensor2d(tokens.map((d) => d.vector));
  const vectorNorms = tf.norm(
    vectors,
    /*ord=*/ 2,
    /*dim=*/ 1,
    /*keep_dims=*/ true
  ) as tf.Tensor2D;
  const vectorsNormed = tf.div(vectors, vectorNorms) as tf.Tensor2D;
  writeNormedVectorsToTokens(tokens, vectorNorms, vectorsNormed);

  return new VectorManager(tokens, vectors, vectorNorms, vectorsNormed);
}
