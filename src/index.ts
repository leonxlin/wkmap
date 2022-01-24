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
  vectorNormed?: number[];
  freqRank: number;
  node?: Element;
  plotPos: number[];
}

async function getData(): Promise<Record[]> {
  const raw = await d3.text("./data/wiki-news-300d-10k-filtered.vec");
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

function saveVectorsNormed(data: Record[], vectorsNormed: tf.Tensor) {
  const vals = vectorsNormed.arraySync() as number[][];
  for (let i = 0; i < vals.length; ++i) {
    data[i].vectorNormed = vals[i];
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
    });
}

function useGirlBoyPositions(data: Record[], vectors: tf.Tensor) {
  let girl!: tf.Tensor, boy!: tf.Tensor;
  data.forEach((d) => {
    if (d.word == "girl" && d.vectorNormed) {
      girl = tf.tensor(d.vectorNormed);
    } else if (d.word == "boy" && d.vectorNormed) {
      boy = tf.tensor(d.vectorNormed);
    }
  });

  const N = data.length;

  if (!girl || !boy) {
    console.log("One or both vectors not found!");
    return;
  }

  const girlBoy = tf.sub(boy, girl);
  const girlOthers = tf.sub(vectors, tf.reshape(girl, [1, 300]));
  const sims = tf.div(
    tf.matMul(girlOthers, tf.reshape(girlBoy, [300, 1])),
    tf.dot(girlBoy, girlBoy)
  );
  const distancesToGirlBoyLine = tf.norm(
    tf.sub(
      girlOthers,
      tf.matMul(tf.reshape(sims, [N, 1]), tf.reshape(girlBoy, [1, 300]))
    ),
    /*ord=*/ 2,
    /*axis=*/ 1
  );

  const distancesArr = tf.util.flatten(
    distancesToGirlBoyLine.arraySync()
  ) as number[];
  const simsArr = tf.util.flatten(sims.arraySync()) as number[];

  updatePositions(
    data,
    (d: Record) => simsArr[d.freqRank],
    (d: Record) => distancesArr[d.freqRank]
  );
}

getData().then(function (data: Record[]) {
  data = data.slice(0, 10000);

  const vectors = tf.tensor2d(data.map((d) => d.vector));
  const vectorsNormed = tf.div(
    vectors,
    tf.norm(vectors, /*ord=*/ 2, /*dim=*/ 0, /*keepDims=*/ true)
  );
  saveVectorsNormed(data, vectorsNormed);

  d3.selectAll<HTMLInputElement, undefined>("[name=projection]").on(
    "click",
    function () {
      if (this.value == "comp01") {
        updatePositions(data, getComponent(0), getComponent(1));
      } else if (this.value == "comp23") {
        updatePositions(data, getComponent(2), getComponent(3));
      } else if (this.value == "girlboy") {
        useGirlBoyPositions(data, vectorsNormed);
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
