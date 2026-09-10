import nextJest from "next/jest.js";

// next/jest wires up SWC, the `@/*` path alias, CSS handling and NEXT_PUBLIC_*
// env loading, so these tests compile the same TSX the app ships.
// The config is .mjs rather than .ts on purpose: a TypeScript jest config
// would pull in ts-node solely to read this file.
const createJestConfig = nextJest({ dir: "./" });

/** @type {import('jest').Config} */
const config = {
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  testPathIgnorePatterns: ["/node_modules/", "/.next/"],
  // The standalone build copies package.json, which collides with the real
  // one in Jest's module map.
  modulePathIgnorePatterns: ["<rootDir>/.next/"],
  // fixtures.ts holds builders, not tests.
  testMatch: ["<rootDir>/__tests__/**/*.test.{ts,tsx}"],
  collectCoverageFrom: ["components/**/*.tsx", "lib/**/*.ts"],
};

export default createJestConfig(config);
