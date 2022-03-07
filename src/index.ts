import * as d3 from "d3";

import { Token } from "./token";
import { VectorManager, loadVectors } from "./vector";

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

function updatePositions(data: Token[]): void {
  const getX = (d: Token) => d.plotPos[0];
  const getY = (d: Token) => d.plotPos[1];

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
    .attr("cx", function (d) {
      return axisX(getX(d as Token));
    })
    .attr("cy", function (d) {
      return axisY(getY(d as Token));
    })
    .filter((d) => {
      return (d as Token).index < 1000;
    })
    .style("display", "inline");
}

function showWords(words: string[], highlight = false): void {
  const selection = d3
    .selectAll(".embedding")
    .filter((d) => {
      return words.includes((d as Token).name);
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

d3.select<HTMLInputElement, undefined>("[name=customWord]").on(
  "input",
  function () {
    showWords([this.value], true);
  }
);

// TODO: Make this DRYer.
function useWordPairAxis(
  manager: VectorManager,
  wordsA: string[],
  wordsB: string[]
) {
  manager.projectToTokenPairsAxis(wordsA, wordsB);
  updatePositions(manager.tokens);
  showWords(wordsA.concat(wordsB), /*highlight=*/ true);
}

function useWordPairAxis2(
  manager: VectorManager,
  wordsA: string[],
  wordsB: string[]
) {
  manager.projectToTokenPairsAxis2(wordsA, wordsB);
  updatePositions(manager.tokens);
  showWords(wordsA.concat(wordsB), /*highlight=*/ true);
}

function createDots(manager: VectorManager) {
  if (!manager.tokens) return;

  const indexToRadius = d3
    .scalePow()
    .exponent(0.5)
    .domain(d3.extent(manager.tokens, (d) => d.index) as [number, number])
    .range([15, 2]);

  svg.select("g").remove();
  svg
    .append("g")
    .selectAll("dot")
    .data(manager.tokens)
    .join("circle")
    .attr("class", "embedding")
    .attr("r", (d) => indexToRadius(d.index))
    .style("fill", defaultColor)
    .style("opacity", 0.5)
    .style("display", "none")
    .on("mouseover", function (event: MouseEvent, d: Token) {
      tooltip.style("display", "block");
    })
    .on("mousemove", function (this: d3.BaseType, event: MouseEvent, d: Token) {
      const coords = d3.pointer(event, svg.node());
      coords[0] += 10;
      coords[1] += 10;
      tooltip
        .html(`${d.name}: ${d.plotPos}`)
        .style("left", coords[0] + "px")
        .style("top", coords[1] + "px");
      d3.select(this).style("opacity", 1);
    })
    .on(
      "mouseleave",
      function (this: d3.BaseType, event: MouseEvent, d: Token) {
        d3.select(this).style("opacity", 0.5);
        tooltip.style("display", "none");
      }
    )
    .on("click", (event: MouseEvent, d: Token) => {
      // Highlight the 10 nearest neighbors.

      d3.selectAll(".embedding").style("fill", defaultColor);

      const similarTokens = manager.getNeighbors(d);
      showWords(
        similarTokens.map((d) => d.name),
        /*highlight=*/ true
      );
    });
}

(() => {
  let manager: VectorManager;

  d3.select<HTMLSelectElement, undefined>("[name=data-sources]").on(
    "change",
    function () {
      // TODO: make this DRYer.
      const selectedValue = d3.select(this).property("value");
      if (selectedValue == "./data/wiki-news-300d-50k-filtered.vec") {
        loadVectors(selectedValue).then((m) => {
          manager = m;
          createDots(manager);
          updatePositions(manager.projectToComponents(0, 1));
        });
      } else if (
        selectedValue ==
        "./data/wikipedia2vec_enwiki_20180420_300d_un_members.txt"
      ) {
        loadVectors(selectedValue, /*skipHeader=*/ true).then((m) => {
          manager = m;
          createDots(manager);
          updatePositions(manager.projectToComponents(0, 1));
        });
      }
    }
  );

  d3.selectAll<HTMLInputElement, undefined>("[name=projection]").on(
    "click",
    function () {
      if (this.value == "comp01") {
        updatePositions(manager.projectToComponents(0, 1));
      } else if (this.value == "comp23") {
        updatePositions(manager.projectToComponents(2, 3));
      } else if (this.value == "girlboy") {
        useWordPairAxis(manager, ["girl"], ["boy"]);
      } else if (this.value == "freqlen") {
        updatePositions(
          manager.setTokenPlotPositionsFromFn(
            (d) => Math.log(d.index + 1),
            (d) => d.norm || -1
          )
        );
      } else if (this.value == "gender") {
        useWordPairAxis(
          manager,
          ["girl", "woman", "female", "she", "herself", "mother", "daughter"],
          ["boy", "man", "male", "he", "himself", "father", "son"]
        );
      } else if (this.value == "liberty") {
        useWordPairAxis(
          manager,
          ["libertarian", "liberty", "libertarianism"],
          ["authoritarian", "authority", "authoritarianism"]
        );
      } else if (this.value == "gender2") {
        useWordPairAxis2(
          manager,
          ["girl", "woman", "female", "she", "herself", "mother", "daughter"],
          ["boy", "man", "male", "he", "himself", "father", "son"]
        );
      } else if (this.value == "liberty2") {
        useWordPairAxis2(
          manager,
          ["libertarian", "liberty", "libertarianism"],
          ["authoritarian", "authority", "authoritarianism"]
        );
      } else if (this.value == "chinaus") {
        useWordPairAxis2(manager, ["China", "Chinese"], ["U.S.", "American"]);
      } else if (this.value == "chinaus_entities") {
        useWordPairAxis(manager, ["ENTITY/United_States"], ["ENTITY/China"]);
      }
    }
  );
})();
