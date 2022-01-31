import * as d3 from "d3";

import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";

/* eslint-disable @typescript-eslint/no-explicit-any */
// Needed to make typescript happy when defining properties on the global window object for easy debugging.
declare global {
  interface Window {
    d3: any;
    tf: any;
  }
}
window.d3 = d3;
window.tf = tf;
/* eslint-enable @typescript-eslint/no-explicit-any */

// Set the dimensions and margins of the plot
const margin = { top: 30, right: 30, bottom: 30, left: 60 },
  width = 800,
  height = 800;

// Append the svg object to the body of the page
const svg = d3
  .select(".plot svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom);

const tooltip = d3.select(".tooltip");

interface Record {
  word: string;
  vector: number[];
  norm?: number;
  vectorNormed?: number[];

  // Rank by word frequency (0 corresponding to most frequent). Also, index into
  freqRank: number;
  node?: Element;
  plotPos: number[];
}

async function getData(): Promise<Record[]> {
  const raw = await d3.text("./data/wiki-news-300d-50k-filtered.vec");
  const dsv = d3.dsvFormat(" ");
  return dsv.parseRows(raw).map((row, index) => {
    return {
      word: row[0],
      vector: row.slice(1).map((a) => parseFloat(a)),
      freqRank: index,
      plotPos: [0, 0],
    };
  });
}

function saveVectorsNormed(
  data: Record[],
  vectorNorms: tf.Tensor,
  vectorsNormed: tf.Tensor
) {
  const vnds = vectorsNormed.arraySync() as number[][];
  const vns = tf.util.flatten(vectorNorms.arraySync()) as number[];
  for (let i = 0; i < vnds.length; ++i) {
    data[i].vectorNormed = vnds[i];
    data[i].norm = vns[i];
  }
}

function getComponent(i: number) {
  return (d: Record) => (d.vectorNormed ? d.vectorNormed[i] : d.vector[i]);
}

function updatePositions(
  data: Record[],
  getX: (d: Record) => number,
  getY: (d: Record) => number
): void {
  // TODO: consider doing transforms within svg instead of in d3?
  // Add X axis
  const axisX = d3
    .scaleLinear()
    .domain(d3.extent(data, getX) as [number, number])
    .range([margin.left, margin.left + width]);
  d3.select(".x-axis").remove();
  svg
    .append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0, ${height + margin.top})`)
    .call(d3.axisBottom(axisX));

  // Add Y axis
  const axisY = d3
    .scaleLinear()
    .domain(d3.extent(data, getY) as [number, number])
    .range([margin.top + height, margin.top]);
  d3.select(".y-axis").remove();
  svg
    .append("g")
    .attr("class", "y-axis")
    .attr("transform", `translate(${margin.left}, 0)`)
    .call(d3.axisLeft(axisY));

  d3.selectAll(".word-embedding")
    .each((d) => {
      const dr = d as Record;
      dr.plotPos = [getX(dr), getY(dr)];
    })
    .attr("cx", function (d) {
      return axisX(getX(d as Record));
    })
    .attr("cy", function (d) {
      return axisY(getY(d as Record));
    })
    .filter((d) => {
      return (d as Record).freqRank < 1000;
    })
    .style("display", "inline");
}

function showWords(words: string[]): void {
  d3.selectAll(".word-embedding")
    .filter((d) => {
      return words.includes((d as Record).word);
    })
    .style("display", "inline");
}

function computeWordPairProjection(
  wordA: string,
  wordB: string,
  data: Record[],
  vectors: tf.Tensor2D
): tf.Tensor2D {
  let vecA!: tf.Tensor, vecB!: tf.Tensor;
  data.forEach((d) => {
    if (d.word == wordA && d.vectorNormed) {
      vecA = tf.tensor(d.vectorNormed);
    } else if (d.word == wordB && d.vectorNormed) {
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

// wordsA and wordsB need not have the same length.
function useAvgOfWordPairPostions2(
  data: Record[],
  vectors: tf.Tensor2D,
  wordsA: string[],
  wordsB: string[]
) {
  const vecsA = [] as tf.Tensor2D[],
    vecsB = [] as tf.Tensor2D[];
  data.forEach((d) => {
    if (wordsA.includes(d.word) && d.vectorNormed) {
      vecsA.push(tf.tensor2d(d.vectorNormed, [300, 1]));
    } else if (wordsB.includes(d.word) && d.vectorNormed) {
      vecsB.push(tf.tensor2d(d.vectorNormed, [300, 1]));
    }
  });

  const matA = tf.concat2d(vecsA, /*axis=*/ 1);
  const matB = tf.concat2d(vecsB, /*axis=*/ 1);
  const simsA = tf.matMul(vectors, matA);
  const simsB = tf.matMul(vectors, matB);
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
  const abScale = tf.div(aggDistA, totDist);
  const coords = tf.concat([abScale, totDist], /*axis=*/ 1);
  const coordsArr = coords.arraySync() as number[][];

  updatePositions(
    data,
    (d: Record) => coordsArr[d.freqRank][0],
    (d: Record) => coordsArr[d.freqRank][1]
  );
  showWords(wordsA.concat(wordsB));
}

// wordsA and wordsB must be the same length.
function useAvgOfWordPairPositions(
  data: Record[],
  vectors: tf.Tensor2D,
  wordsA: string[],
  wordsB: string[]
) {
  const projs: tf.Tensor2D[] = [];
  for (let i = 0; i < wordsA.length; ++i) {
    projs.push(computeWordPairProjection(wordsA[i], wordsB[i], data, vectors));
  }
  const coords = tf.div(tf.addN(projs), wordsA.length) as tf.Tensor2D;
  const coordsArr = coords.arraySync() as number[][];

  updatePositions(
    data,
    (d: Record) => coordsArr[d.freqRank][0],
    (d: Record) => coordsArr[d.freqRank][1]
  );
  showWords(wordsA.concat(wordsB));
}

function useGirlBoyPositions(data: Record[], vectors: tf.Tensor2D) {
  const coords = computeWordPairProjection("girl", "boy", data, vectors);
  const coordsArr = coords.arraySync() as number[][];

  updatePositions(
    data,
    (d: Record) => coordsArr[d.freqRank][0],
    (d: Record) => coordsArr[d.freqRank][1]
  );
  showWords(["girl", "boy"]);
}

getData().then(function (data: Record[]) {
  const vectors = tf.tensor2d(data.map((d) => d.vector));
  const vectorNorms = tf.norm(
    vectors,
    /*ord=*/ 2,
    /*dim=*/ 1,
    /*keep_dims=*/ true
  );
  const vectorsNormed = tf.div(vectors, vectorNorms) as tf.Tensor2D;
  saveVectorsNormed(data, vectorNorms, vectorsNormed);

  d3.selectAll<HTMLInputElement, undefined>("[name=projection]").on(
    "click",
    function () {
      if (this.value == "comp01") {
        updatePositions(data, getComponent(0), getComponent(1));
      } else if (this.value == "comp23") {
        updatePositions(data, getComponent(2), getComponent(3));
      } else if (this.value == "girlboy") {
        useGirlBoyPositions(data, vectorsNormed);
      } else if (this.value == "freqlen") {
        updatePositions(
          data,
          (d) => Math.log(d.freqRank + 1),
          (d) => d.norm || -1
        );
      } else if (this.value == "gender") {
        useAvgOfWordPairPositions(
          data,
          vectorsNormed,
          ["girl", "woman", "female", "she"],
          ["boy", "man", "male", "he"]
        );
      } else if (this.value == "liberty") {
        useAvgOfWordPairPositions(
          data,
          vectorsNormed,
          ["libertarian", "liberty", "libertarianism"],
          ["authoritarian", "authority", "authoritarianism"]
        );
      } else if (this.value == "gender2") {
        useAvgOfWordPairPostions2(
          data,
          vectorsNormed,
          ["girl", "woman", "female", "she"],
          ["boy", "man", "male", "he"]
        );
      } else if (this.value == "liberty2") {
        useAvgOfWordPairPostions2(
          data,
          vectorsNormed,
          ["libertarian", "liberty", "libertarianism"],
          ["authoritarian", "authority", "authoritarianism"]
        );
      }
    }
  );

  const defaultColor = "#69b3a2";

  const freqRankToRadius = d3
    .scalePow()
    .exponent(0.5)
    .domain(d3.extent(data, (d) => d.freqRank) as [number, number])
    .range([15, 2]);

  // Add dots
  svg
    .append("g")
    .selectAll("dot")
    .data(data)
    .join("circle")
    .attr("class", "word-embedding")
    .attr("r", (d) => freqRankToRadius(d.freqRank))
    .style("fill", defaultColor)
    .style("opacity", 0.5)
    .style("display", "none")
    .on("mouseover", function (event: MouseEvent, d: Record) {
      tooltip.style("display", "block");
    })
    .on(
      "mousemove",
      function (this: d3.BaseType, event: MouseEvent, d: Record) {
        const coords = d3.pointer(event, svg.node());
        coords[0] += 10;
        coords[1] += 10;
        tooltip
          .html(`${d.word}: ${d.plotPos}`)
          .style("left", coords[0] + "px")
          .style("top", coords[1] + "px");
        d3.select(this).style("opacity", 1);
      }
    )
    .on(
      "mouseleave",
      function (this: d3.BaseType, event: MouseEvent, d: Record) {
        d3.select(this).style("opacity", 0.5);
        tooltip.style("display", "none");
      }
    )
    .on("click", (event: MouseEvent, d: Record) => {
      // Highlight the 10 nearest neighbors.

      d3.selectAll(".word-embedding").style("fill", defaultColor);

      const similarities = tf.util.flatten(
        tf
          .matMul(
            vectorsNormed,
            tf.tensor2d(d.vectorNormed || d.vector, [300, 1])
          )
          .arraySync()
      ) as number[];
      const sim10 = [...similarities].sort(d3.descending)[10];

      d3.selectAll(".word-embedding")
        .filter((d) => similarities[(d as Record).freqRank] >= sim10)
        .style("display", "inline")
        .style("fill", "red")
        .each(function (this) {
          const node = this as Element;
          if (node.parentNode) {
            node.parentNode.appendChild(node); // Bring to front.
          }
        });
    });

  updatePositions(data, getComponent(0), getComponent(1));
});
