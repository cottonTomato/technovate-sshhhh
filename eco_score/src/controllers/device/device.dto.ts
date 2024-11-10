import { z } from 'zod';

export const getDevicesDto = z.object({});

export type IGetDevicesDto = z.infer<typeof getDevicesDto>;

export const addDeviceDto = z.object({
  type: z.number(),
  name: z.string(),
});

export type IAddDeviceDto = z.infer<typeof addDeviceDto>;

export const getDeviceDataDto = z.object({
  name: z.string(),
});

export type IGetDeviceDataDto = z.infer<typeof getDeviceDataDto>;
