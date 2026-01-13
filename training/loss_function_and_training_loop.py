import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import Point

def distance_circle(points, outputs):
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)
    gdf['geometry'] = gdf.buffer(500)
    gdf = gdf.to_crs(epsg=4326)
    
    distance_matrix = np.zeros((len(gdf), 1))
    for i in range(len(gdf)):
        distance = gdf.iloc[i]['geometry'].distance(Point(outputs[i]))
        distance_matrix[i, :] = distance
    
    return torch.sum(torch.tensor(distance_matrix, dtype=torch.float32).to(device))


def calc_total_loss(criterion, outputs, batch_y, points):
    org_loss = criterion(outputs, batch_y)
    new_loss = distance_circle(points, outputs)
    #print(0.9*org_loss, 0.1*new_loss)
    total_loss = 0.7*org_loss + 0.3*new_loss
    return total_loss


total_epochs = 50
counter = 0
for epoch in range(total_epochs):
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        counter+=1
        model.train()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y[0]) 
        #points = [Point(lon, lat) for lon, lat in batch_X[0,:,-1,:2]]
        #loss = calc_total_loss(criterion, outputs, batch_y, points)
        
        optimizer.zero_grad()
        loss.backward()
        
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        #print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}')        
        if counter%4==0:
            batch_X, batch_y = test_generator.__getitem__(np.random.randint(0, test_generator.__len__()))
            batch_X = batch_X.reshape(1,batch_X.shape[0], batch_X.shape[1],batch_X.shape[2]).to(device)
            batch_y = batch_y.to(device)
            model.eval()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            #points = [Point(lon, lat) for lon, lat in batch_X[0,:,-1,:2]]
            #loss = calc_total_loss(criterion, outputs, batch_y, points)
            print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}') 
        #break
