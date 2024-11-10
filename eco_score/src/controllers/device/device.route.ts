import { Router } from 'express';
import { ClerkExpressRequireAuth } from '@clerk/clerk-sdk-node';
import { validatorFactory } from '../../middlewares/validator.middleware';
import { getDevicesDto, addDeviceDto, getDeviceDataDto } from './device.dto';
import {
  getDevicesHandler,
  addDeviceHandler,
  getDeviceDataHandler,
} from './device.controller';

export const deviceRouter = Router();

deviceRouter.all('/', ClerkExpressRequireAuth());

deviceRouter.get('/', validatorFactory(getDevicesDto), getDevicesHandler);
deviceRouter.post('/', validatorFactory(addDeviceDto), addDeviceHandler);

deviceRouter.get(
  '/data',
  validatorFactory(getDeviceDataDto),
  getDeviceDataHandler
);
