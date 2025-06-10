const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  entry: {
    popup: "./src/popup/index.tsx",
    content: "./src/content/index.ts",
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "[name].js",
  },  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
      {
        test: /\.(woff|woff2|eot|ttf|otf)$/i,
        type: 'asset/resource',
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },  plugins: [
    new CopyPlugin({
      patterns: [
        { from: "manifest.json" },
        { from: "popup.html" },
        { from: "src/popup/popup.css", to: "popup.css" },
        { from: "icons", to: "icons" },
      ],
    }),
  ],
  optimization: {
    minimize: false, // Don't minify the content script for better debugging
  },
};
