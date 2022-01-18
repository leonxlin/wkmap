import resolve from "@rollup/plugin-node-resolve";
import typescript from "@rollup/plugin-typescript";

export default {
  input: "src/index.ts",
  output: [
    {
      file: "bundle.js",
      format: "iife",
      name: "wkmap",
    },
  ],
  plugins: [typescript(), resolve()],
};
