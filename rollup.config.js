import typescript from "@rollup/plugin-typescript";
import peerDepsExternal from "rollup-plugin-peer-deps-external";
import dts from "rollup-plugin-dts";

export default [
  {
    input: "src/index.ts",
    output: {
      file: "./build/index.mjs",
      format: "es",
    },
    plugins: [
      peerDepsExternal(),
      typescript({
        tsconfig: "./tsconfig.json",
        outDir: "./build",
        declaration: false,
        declarationMap: false,
        sourceMap: false,
      }),
    ],
  },
  {
    input: "src/index.ts",
    output: {
      file: "./build/index.cjs",
      format: "cjs",
    },
    plugins: [
      peerDepsExternal(),
      typescript({
        tsconfig: "./tsconfig.json",
        outDir: "./build",
        declaration: false,
        declarationMap: false,
        sourceMap: false,
      }),
    ],
  },
  {
    input: "src/index.ts",
    output: {
      file: "./types.d.ts",
      format: "es",
    },
    plugins: [dts()],
  },
];
