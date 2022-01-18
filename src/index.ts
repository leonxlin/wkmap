import { add } from "./math";

import * as d3 from "d3";

const xx = 20;
const yy = 10;

console.log(`${xx} + ${yy} = ${add(xx, yy)}`);
// set the dimensions and margins of the graph
const margin = { top: 10, right: 30, bottom: 30, left: 60 },
  width = 460 - margin.left - margin.right,
  height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
const svg = d3
  .select(".plot")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", `translate(${margin.left}, ${margin.top})`);

interface Record {
  word: string;
  vector: number[];
}

async function getData(): Promise<Record[]> {
  const raw = await d3.text("./data/wiki-news-300d-10k-filtered.vec");
  const dsv = d3.dsvFormat(" ");
  return dsv.parseRows(raw).map((row) => {
    return {
      word: row[0],
      vector: row.slice(1).map((a) => parseFloat(a)),
    };
  });
}

getData().then(function (data: Record[]) {
  console.log(data.slice(0, 100));

  // Add X axis
  const x = d3
    .scaleLinear()
    .domain(d3.extent(data, (d) => d.vector[0]) as [number, number])
    .range([0, width]);
  svg
    .append("g")
    .attr("transform", `translate(0, ${height})`)
    .call(d3.axisBottom(x));

  // Add Y axis
  const y = d3
    .scaleLinear()
    .domain(d3.extent(data, (d) => d.vector[1]) as [number, number])
    .range([0, width]);
  svg.append("g").call(d3.axisLeft(y));

  console.log(x);
  console.log(y);
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
    .attr("r", 1.5)
    .style("fill", "#69b3a2");
});