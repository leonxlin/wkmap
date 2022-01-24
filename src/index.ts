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
const margin = { top: 0, right: 0, bottom: 30, left: 60 },
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
  vectorNormed?: number[];
  freqRank: number;
  node?: Element;
}

async function getData(): Promise<Record[]> {
  const raw = await d3.text("./data/wiki-news-300d-10k-filtered.vec");
  const dsv = d3.dsvFormat(" ");
  return dsv.parseRows(raw).map((row, index) => {
    return {
      word: row[0],
      vector: row.slice(1).map((a) => parseFloat(a)),
      freqRank: index,
    };
  });
}

function saveVectorsNormed(data: Record[], vectorsNormed: tf.Tensor) {
  const vals = vectorsNormed.arraySync() as number[][];
  for (let i = 0; i < vals.length; ++i) {
    data[i].vectorNormed = vals[i];
  }
}

function xComp(d: Record) {
  // TODO: make this better
  return d.vectorNormed ? d.vectorNormed[0] : d.vector[0];
}

function yComp(d: Record) {
  // TODO: make this better
  return d.vectorNormed ? d.vectorNormed[1] : d.vector[1];
}

getData().then(function (data: Record[]) {
  data = data.slice(0, 10000);

  const vectors = tf.tensor2d(data.map((d) => d.vector));
  const vectorsNormed = tf.div(
    vectors,
    tf.norm(vectors, /*ord=*/ 2, /*dim=*/ 0, /*keepDims=*/ true)
  );
  saveVectorsNormed(data, vectorsNormed);

  const defaultColor = "#69b3a2";

  // TODO: consider doing transforms within svg instead of in d3?
  // Add X axis
  const x = d3
    .scaleLinear()
    .domain(d3.extent(data, xComp) as [number, number])
    .range([margin.left, margin.left + width]);
  svg
    .append("g")
    .attr("transform", `translate(0, ${height})`)
    .call(d3.axisBottom(x));

  // Add Y axis
  const y = d3
    .scaleLinear()
    .domain(d3.extent(data, yComp) as [number, number])
    .range([margin.top + height, margin.top]);
  svg
    .append("g")
    .attr("transform", `translate(${margin.left}, 0)`)
    .call(d3.axisLeft(y));

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
    .attr("cx", function (d) {
      return x(xComp(d));
    })
    .attr("cy", function (d) {
      return y(yComp(d));
    })
    .attr("r", (d) => freqRankToRadius(d.freqRank))
    .style("fill", defaultColor)
    .style("opacity", 0.5)
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
          .html(`${d.word}: ${xComp(d)}, ${yComp(d)}`)
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
        .style("fill", "red")
        .each(function (this) {
          const node = this as Element;
          if (node.parentNode) {
            node.parentNode.appendChild(node); // Bring to front.
          }
        });
    });
});
