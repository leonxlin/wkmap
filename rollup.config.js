import resolve from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";
import commonjs from "@rollup/plugin-commonjs";

export default [
  {
    input: ["src/index.ts"],
    output: [
      {
        file: "bundle.js",
        format: "iife",
        name: "wkmap",
      },
    ],
    plugins: [typescript(), commonjs(), resolve()],
  },
  {
    input: ["src/countries.ts"],
    output: [
      {
        file: "countries_bundle.js",
        format: "iife",
        name: "countries_bundle",
      },
    ],
    plugins: [typescript(), commonjs(), resolve()],
  },
];
