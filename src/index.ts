import * as d3 from "d3";

import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";

/* eslint-disable @typescript-eslint/no-explicit-any */
// Needed to make typescript happy when defining properties on the global window object for easy debugging.
declare global {
  interface Window {
    d3: any;
  }
}
window.d3 = d3;
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
  freqRank: number;
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

getData().then(function (data: Record[]) {
  data = data.slice(0, 1000);

  console.log(tf.tensor(data[0].vector));

  // TODO: consider doing transforms within svg instead of in d3?
  // Add X axis
  const x = d3
    .scaleLinear()
    .domain(d3.extent(data, (d) => d.vector[0]) as [number, number])
    .range([margin.left, margin.left + width]);
  svg
    .append("g")
    .attr("transform", `translate(0, ${height})`)
    .call(d3.axisBottom(x));

  // Add Y axis
  const y = d3
    .scaleLinear()
    .domain(d3.extent(data, (d) => d.vector[1]) as [number, number])
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
    .data(data.slice(0, 1000))
    .join("circle")
    .attr("cx", function (d) {
      return x(d.vector[0]);
    })
    .attr("cy", function (d) {
      return y(d.vector[1]);
    })
    .attr("r", (d) => freqRankToRadius(d.freqRank))
    .style("fill", "#69b3a2")
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
          .html(`${d.word}: ${d.vector[0]}, ${d.vector[1]}`)
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
    );
});
