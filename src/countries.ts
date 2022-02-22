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

const defaultColor = "#69b3a2";

// Append the svg object to the body of the page
const svg = d3
  .select(".plot svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom);

const tooltip = d3.select(".tooltip");

interface Record {
  name: string;
  vector: number[];
  norm?: number;
  vectorNormed?: number[];

  // Rank by entity frequency (0 corresponding to most frequent). Also, index into output of getData().
  freqRank: number;
  node?: Element;
  plotPos: number[];
}

async function getData(): Promise<Record[]> {
  const raw = await d3.text(
    "./data/wikipedia2vec_enwiki_20180420_300d_un_members.txt"
  );
  const dsv = d3.dsvFormat(" ");
  return dsv
    .parseRows(raw)
    .slice(1)
    .map((row, index) => {
      return {
        name: row[0],
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

  d3.selectAll(".embedding")
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

function showNames(names: string[], highlight = false): void {
  const selection = d3
    .selectAll(".embedding")
    .filter((d) => {
      return names.includes((d as Record).name);
    })
    .style("display", "inline");

  if (highlight) {
    d3.selectAll(".embedding").style("fill", defaultColor);
    selection.style("fill", "red").each(function (this) {
      const node = this as Element;
      if (node.parentNode) {
        node.parentNode.appendChild(node); // Bring to front.
      }
    });
  }
}

function computePairProjection(
  nameA: string,
  nameB: string,
  data: Record[],
  vectors: tf.Tensor2D
): tf.Tensor2D {
  let vecA!: tf.Tensor, vecB!: tf.Tensor;
  data.forEach((d) => {
    if (d.name == nameA && d.vectorNormed) {
      vecA = tf.tensor(d.vectorNormed);
    } else if (d.name == nameB && d.vectorNormed) {
      vecB = tf.tensor(d.vectorNormed);
    }
  });

  const N = data.length;

  if (!vecA || !vecB) {
    throw new Error(`Vector not found for ${nameA} or ${nameB} or both`);
    console.log(`Vector not found for ${nameA} or ${nameB} or both`);
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

function useUSChinaPositions(data: Record[], vectors: tf.Tensor2D) {
  const coords = computePairProjection(
    "ENTITY/United_States",
    "ENTITY/China",
    data,
    vectors
  );
  const coordsArr = coords.arraySync() as number[][];

  updatePositions(
    data,
    (d: Record) => coordsArr[d.freqRank][0],
    (d: Record) => coordsArr[d.freqRank][1]
  );
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
      } else if (this.value == "uschina") {
        useUSChinaPositions(data, vectorsNormed);
      } else if (this.value == "freqlen") {
        updatePositions(
          data,
          (d) => Math.log(d.freqRank + 1),
          (d) => d.norm || -1
        );
      }
    }
  );

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
    .attr("class", "embedding")
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
          .html(`${d.name}: ${d.plotPos}`)
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

      d3.selectAll(".embedding").style("fill", defaultColor);

      const similarities = tf.util.flatten(
        tf
          .matMul(
            vectorsNormed,
            tf.tensor2d(d.vectorNormed || d.vector, [300, 1])
          )
          .arraySync()
      ) as number[];
      const sim10 = [...similarities].sort(d3.descending)[10];

      showNames(
        data
          .filter((d) => similarities[d.freqRank] >= sim10)
          .map((d) => d.name),
        /*highlight=*/ true
      );
    });

  updatePositions(data, getComponent(0), getComponent(1));

  d3.select<HTMLInputElement, undefined>("[name=customName]").on(
    "input",
    function () {
      showNames([this.value], true);
    }
  );
});
