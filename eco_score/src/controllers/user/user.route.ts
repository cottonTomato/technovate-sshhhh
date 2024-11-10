import { Router } from 'express';
import { ClerkExpressRequireAuth } from '@clerk/clerk-sdk-node';
import { validatorFactory } from '../../middlewares/validator.middleware';
import { addUserDto } from './user.dto';
import { addUserHandler } from './user.controller';

export const userRouter = Router();

userRouter.post(
  '/',
  ClerkExpressRequireAuth(),
  validatorFactory(addUserDto),
  addUserHandler
);
