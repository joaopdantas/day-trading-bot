import React, { useEffect, useRef } from "react";
import { createChart, ColorType, Time, UTCTimestamp } from "lightweight-charts";
import { MarketData } from "../types";

interface ChartProps {
  data: MarketData[];
  width?: number;
  height?: number;
}

const MarketChart: React.FC<ChartProps> = ({
  data,
  width = 600,
  height = 400,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chart = useRef<ReturnType<typeof createChart> | null>(null);

  // Debug data changes
  useEffect(() => {
    console.log("MarketChart received data:", {
      dataLength: data.length,
      firstPoint: data[0],
      lastPoint: data[data.length - 1],
      containerWidth: chartContainerRef.current?.clientWidth,
      containerHeight: chartContainerRef.current?.clientHeight,
    });
  }, [data]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) {
      console.log("Chart container ref is not available");
      return;
    }

    // Get the actual container width
    const containerWidth = chartContainerRef.current.clientWidth || width;
    console.log("Initializing chart with dimensions:", { containerWidth, height });

    chart.current = createChart(chartContainerRef.current, {
      width: containerWidth,
      height,
      layout: {
        background: { color: "#ffffff" },
        textColor: "#333",
      },
      grid: {
        vertLines: { color: "#f0f0f0" },
        horzLines: { color: "#f0f0f0" },
      },
      rightPriceScale: {
        borderVisible: true,
        borderColor: "#d6dcde",
      },
      timeScale: {
        borderVisible: true,
        borderColor: "#d6dcde",
        timeVisible: true,
      },
    });

    const candleSeries = chart.current.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
      borderVisible: false,
    });

    const volumeSeries = chart.current.addHistogramSeries({
      color: "#26a69a",
      priceScaleId: "volume",
      priceFormat: {
        type: "volume",
      },
    });

    // Set the price scale properties for the volume series
    chart.current.priceScale("volume").applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
      alignLabels: true,
    });

    // Handle resizing
    const handleResize = () => {
      if (chartContainerRef.current && chart.current) {
        const { clientWidth, clientHeight } = chartContainerRef.current;
        chart.current.applyOptions({
          width: clientWidth,
          height: clientHeight || height,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    // Transform the data
    if (data.length > 0) {
      console.log("Setting chart data:", data);

      const candleData = data.map((item) => ({
        time: (item.timestamp / 1000) as UTCTimestamp,
        open: item.open || item.price,
        high: item.high || item.price,
        low: item.low || item.price,
        close: item.close || item.price,
      }));

      const volumeData = data.map((item) => ({
        time: (item.timestamp / 1000) as UTCTimestamp,
        value: item.volume,
      }));

      candleSeries.setData(candleData);
      volumeSeries.setData(volumeData);
      chart.current.timeScale().fitContent();
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      if (chart.current) {
        chart.current.remove();
      }
    };
  }, [data, height, width]); // Only recreate chart on size prop changes

  return (
    <div
      ref={chartContainerRef}
      style={{
        width: "100%",
        height,
        borderRadius: "4px",
        border: "1px solid #d6dcde",
        overflow: "hidden",
        backgroundColor: "#ffffff",
      }}
    />
  );
};

export default MarketChart;
