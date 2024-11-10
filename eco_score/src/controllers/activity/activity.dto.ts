import { z } from 'zod';

export const getActivitiesDto = z.object({});

export type IGetActivitiesDto = z.infer<typeof getActivitiesDto>;

export const getActivityDataDto = z.object({
  id: z.number(),
});

export type IGetActivityDataDto = z.infer<typeof getActivityDataDto>;
