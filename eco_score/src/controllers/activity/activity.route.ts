import { Router } from 'express';
import { ClerkExpressRequireAuth } from '@clerk/clerk-sdk-node';
import { validatorFactory } from '../../middlewares/validator.middleware';
import { getActivitiesDto, getActivityDataDto } from './activity.dto';
import {
  getActivityDataHandler,
  getActivitiesHandler,
} from './activity.controller';

export const activityRouter = Router();

activityRouter.all('/', ClerkExpressRequireAuth());

activityRouter.get(
  '/',
  validatorFactory(getActivitiesDto),
  getActivitiesHandler
);

activityRouter.get(
  '/data',
  validatorFactory(getActivityDataDto),
  getActivityDataHandler
);
