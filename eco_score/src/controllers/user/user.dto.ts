import { z } from 'zod';

export const addUserDto = z.object({
  firstname: z.string(),
  lastname: z.string(),
});

export type IAddUserDto = z.infer<typeof addUserDto>;
