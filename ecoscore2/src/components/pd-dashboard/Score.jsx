"use client";

import * as React from "react";
import { TrendingUp } from "lucide-react";
import { Label, Pie, PieChart } from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
const chartData = [
  { day: "Monday", EcoScore: 12, fill: "var(--color-Monday)" },
  { day: "Tuesday", EcoScore: 20, fill: "var(--color-Tuesday)" },
  { day: "Wednesday", EcoScore: 2, fill: "var(--color-Wednesday)" },
  { day: "Thursday", EcoScore: 10, fill: "var(--color-Thursday)" },
  { day: "Friday", EcoScore: 33, fill: "var(--color-Friday)" },
  { day: "Saturday", EcoScore: 7, fill: "var(--color-Saturday)" },
  { day: "Sunday", EcoScore: 3, fill: "var(--color-Saturday)" },
];

const chartConfig = {
  EcoScore: {
    label: "EcoScore",
  },
  Monday: {
    label: "Monday",
    color: "hsl(var(--chart-1))",
  },
  Tuesday: {
    label: "Tuesday",
    color: "hsl(var(--chart-2))",
  },
  Wednesday: {
    label: "Wednesday",
    color: "hsl(var(--chart-3))",
  },
  Thursday: {
    label: "Thursday",
    color: "hsl(var(--chart-4))",
  },
  Friday: {
    label: "Friday",
    color: "hsl(var(--chart-5))",
  },
  Saturday: {
    label: "Saturday",
    color: "hsl(var(--chart-6))",
  },
  Sunday: {
    label: "Sunday",
    color: "hsl(var(--chart-7))",
  },
};

export function Score() {
  const totalEcoScore = React.useMemo(() => {
    return chartData.reduce((acc, curr) => acc + curr.EcoScore, 0);
  }, []);

  return (
    <Card className="flex flex-col h-[50%]">
      <CardHeader className="items-center pb-0">
        <CardTitle>EcoScore</CardTitle>
        <CardDescription>January 2024</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          <PieChart>
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Pie
              data={chartData}
              dataKey="EcoScore"
              nameKey="day"
              innerRadius={60}
              strokeWidth={5}
            >
              <Label
                content={({ viewBox }) => {
                  if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                    return (
                      <text
                        x={viewBox.cx}
                        y={viewBox.cy}
                        textAnchor="middle"
                        dominantBaseline="middle"
                      >
                        <tspan
                          x={viewBox.cx}
                          y={viewBox.cy}
                          className="fill-foreground text-3xl font-bold"
                        >
                          {totalEcoScore.toLocaleString()}
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 24}
                          className="fill-muted-foreground"
                        >
                          EcoScore
                        </tspan>
                      </text>
                    );
                  }
                }}
              />
            </Pie>
          </PieChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col gap-2 text-sm">
        <div className="leading-none text-muted-foreground">
          Your total EcoScore for the last week
        </div>
      </CardFooter>
    </Card>
  );
}
