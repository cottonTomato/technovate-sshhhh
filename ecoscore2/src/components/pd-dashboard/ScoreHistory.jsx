"use client";

import { TrendingUp } from "lucide-react";
import { Bar, BarChart, CartesianGrid, XAxis } from "recharts";

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
  { month: "January", indoor: 186, outdoor: 80 },
  { month: "February", indoor: 305, outdoor: 200 },
  { month: "March", indoor: 237, outdoor: 120 },
  { month: "April", indoor: 73, outdoor: 190 },
  { month: "May", indoor: 209, outdoor: 130 },
  { month: "June", indoor: 214, outdoor: 140 },
];

const chartConfig = {
  indoor: {
    label: "indoor",
    color: "hsl(var(--chart-1))",
  },
  outdoor: {
    label: "outdoor",
    color: "hsl(var(--chart-2))",
  },
};

export function ScoreHistory() {
  return (
    <Card className="h-[50%]">
      <CardHeader>
        <CardTitle>This month</CardTitle>
        <CardDescription>January - June 2024</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <BarChart accessibilityLayer data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="month"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              tickFormatter={(value) => value.slice(0, 3)}
            />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent indicator="dashed" />}
            />
            <Bar dataKey="indoor" fill="var(--color-indoor)" radius={8} />
            <Bar dataKey="outdoor" fill="var(--color-outdoor)" radius={8} />
          </BarChart>
        </ChartContainer>
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        <div className="flex gap-2 font-medium leading-none">
          Trending up by 5.2% this month <TrendingUp className="h-4 w-4" />
        </div>
        <div className="leading-none text-muted-foreground">
          Showing total EcoScore in the last 6 months
        </div>
      </CardFooter>
    </Card>
  );
}
